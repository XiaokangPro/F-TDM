#!/usr/bin/env python3
# rl_extension/run_rl.py
# 强化学习分配系统总入口
#
# 使用方法（在 ftdm_project/ 目录下运行）：
#
#   完整训练（500 episodes）：
#     python rl_extension/run_rl.py
#
#   快速验证（50 episodes，几分钟内出结果）：
#     python rl_extension/run_rl.py --quick
#
#   加载已训练 DQN，跳过训练直接评估：
#     python rl_extension/run_rl.py --load_rl
#
# 前置条件：
#   已运行 main.py 完成 F-TDM 元训练，模型保存在 checkpoints/ftdm_best.pth

import os
import sys
import argparse
import random
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import Config
from data_utils import load_air_quality, build_all_data
from model import TruthDiscoveryNet
from ftdm import FTDM
from rl_extension.grqi import compute_grqi_baseline
from rl_extension.mcs_env import MCSEnv, P_SIZE, K_TOTAL
from rl_extension.rl_train import (
    train_dqn, evaluate_all, print_results,
    N_TRAIN_EPISODES, N_EVAL_EPISODES, RL_SAVE_PATH
)
from rl_extension.dqn_agent import DQNAgent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ftdm(data_path: str):
    """加载预训练 F-TDM 模型和 NO_x 测试数据"""
    print("[1] 加载预训练 F-TDM 模型...")
    if not os.path.exists(Config.SAVE_PATH):
        print(f"  错误：找不到 F-TDM 模型 {Config.SAVE_PATH}")
        print("  请先运行：python main.py --data_path data/AirQualityUCI.csv")
        sys.exit(1)

    model = TruthDiscoveryNet(Config.INPUT_SIZE, Config.HIDDEN_SIZE)
    ftdm  = FTDM(model)
    ftdm.load(Config.SAVE_PATH)
    print(f"  F-TDM 加载成功（隐藏层 {Config.HIDDEN_SIZE}，参数量 {model.count_params():,}）")

    print("[2] 加载 NO_x 测试数据集...")
    df = load_air_quality(data_path)
    _, D_W_test, D_U_test, _ = build_all_data(df)
    print(f"  NO_x 测试集: {D_W_test.shape[0]} 个 PoI，每 PoI {D_W_test.shape[1]} 名参与者")

    return ftdm, D_W_test, D_U_test


def main(args):
    set_seed()
    os.makedirs('checkpoints', exist_ok=True)

    # ── 加载 F-TDM + 数据 ────────────────────────────────────────────
    ftdm, D_W_test, D_U_test = load_ftdm(args.data_path)

    # ── 构建 MCS 环境 ─────────────────────────────────────────────────
    print(f"[3] 构建 MCS 联合分配环境...")
    env = MCSEnv(
        ftdm       = ftdm,
        D_W_pool   = D_W_test,
        D_U_pool   = D_U_test,
        p_size     = P_SIZE,    # P=100 个 PoI
        k_total    = K_TOTAL,   # 预算 20 units
        k_finetune = 1,         # RL 训练快速模式
    )
    print(f"  环境: P={P_SIZE} PoI, Budget={K_TOTAL} units")
    print(f"  状态维度: {env.state_dim}, 动作维度: {env.action_dim}")

    # 计算零分配基线 GRQI（无 UAV，无增强）
    baseline_grqi = compute_grqi_baseline(
        torch.from_numpy(D_W_test[:P_SIZE]),
        D_U_test[:P_SIZE]
    )
    print(f"  零分配基线 GRQI: {baseline_grqi:.4f}")

    # ── 训练 DQN ─────────────────────────────────────────────────────
    n_episodes = 50 if args.quick else N_TRAIN_EPISODES

    if args.load_rl and os.path.exists(RL_SAVE_PATH):
        print(f"\n[4] 加载已训练 DQN: {RL_SAVE_PATH}")
        agent = DQNAgent(env.state_dim, env.action_dim)
        agent.load(RL_SAVE_PATH)
        grqi_log = reward_log = []
    else:
        print(f"\n[4] 训练 DQN ({n_episodes} episodes)...")
        agent, grqi_log, reward_log = train_dqn(env, n_episodes=n_episodes)

    # ── 评估所有方法 ──────────────────────────────────────────────────
    n_eval = 10 if args.quick else N_EVAL_EPISODES
    print(f"\n[5] 评估所有方法（各 {n_eval} episodes）...")
    print("    方法: RL(DQN) / Random / UAV-Only-Random / Greedy-Uncert / Greedy-GRQI")

    # 评估时切换到精确 GRQI（k_finetune=5）
    env.k_ft = 5

    results = evaluate_all(env, agent, n_eval=n_eval)
    print_results(results, baseline_grqi)

    # ── 训练曲线摘要 ──────────────────────────────────────────────────
    if grqi_log:
        print("训练过程 GRQI 变化摘要（每 50 episode 均值）：")
        step = max(1, len(grqi_log) // 10)
        for i in range(0, len(grqi_log), step):
            chunk = grqi_log[i:i+step]
            print(f"  Episode {i+1:4d}~{min(i+step, len(grqi_log)):4d}: "
                  f"GRQI={np.mean(chunk):.4f}")

    print("\n结论：")
    rl_g  = results.get('RL (DQN)', {}).get('grqi_mean', 0)
    rand_g = results.get('Random',  {}).get('grqi_mean', 0)
    if rl_g > rand_g:
        print(f"  RL(DQN) [{rl_g:.4f}] > Random [{rand_g:.4f}]")
        print("  强化学习成功学到了比随机分配更优的策略。")
        print("  GRQI 奖励机制有效驱动智能体向最优数据质量收敛。")
    else:
        print("  提示：训练轮数较少，增加 --episodes 可提升 RL 性能。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRQI 驱动的 RL 联合分配系统')
    parser.add_argument('--data_path', default='data/AirQualityUCI.csv')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（50 episodes），验证代码流程')
    parser.add_argument('--load_rl', action='store_true',
                        help='加载已有 DQN 模型，跳过训练')
    args = parser.parse_args()
    main(args)
