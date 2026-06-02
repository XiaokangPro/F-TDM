# rl_extension/rl_train.py
# RL 训练主循环
#
# 流程：
#   1. 初始化 MCSEnv（加载预训练 F-TDM + NO_x 测试数据）
#   2. 训练 DQN 智能体（500 episodes）
#   3. 对比 5 种方法（RL + 4 基线）各运行 50 episodes 取均值
#   4. 输出结果对比表

import os
import sys
import time
import random
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List

# 把项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import Config
from data_utils import load_air_quality, build_all_data
from model import TruthDiscoveryNet
from ftdm import FTDM
from rl_extension.grqi import compute_grqi_baseline
from rl_extension.mcs_env import MCSEnv, P_SIZE, K_TOTAL
from rl_extension.dqn_agent import DQNAgent
from rl_extension.baselines_rl import (
    run_episode_random,
    run_episode_uav_only,
    run_episode_greedy_uncertainty,
    run_episode_greedy_grqi,
)


# ─────────────────────────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────────────────────────

N_TRAIN_EPISODES   = 500    # 训练 episode 数
N_EVAL_EPISODES    = 30     # 评估每种方法的 episode 数
TARGET_SYNC_EVERY  = 20     # 每 N episode 同步目标网络
LOG_EVERY          = 50     # 日志打印间隔
BATCH_SIZE         = 64
LR                 = 1e-3
GAMMA              = 0.99
BUFFER_SIZE        = 5000
EPS_DECAY          = 400    # ε 线性衰减完成所需 episode 数

RL_SAVE_PATH = 'checkpoints/dqn_mcs.pth'


# ─────────────────────────────────────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(env: MCSEnv, n_episodes: int = N_TRAIN_EPISODES,
              verbose: bool = True) -> DQNAgent:
    """
    DQN 元训练主循环。

    Args:
        env:        MCSEnv 实例（内含预训练 F-TDM）
        n_episodes: 训练 episode 数
        verbose:    是否打印训练进度

    Returns:
        trained DQNAgent
    """
    agent = DQNAgent(
        state_dim  = env.state_dim,
        action_dim = env.action_dim,
        lr         = LR,
        gamma      = GAMMA,
        eps_decay  = EPS_DECAY,
        buffer_size = BUFFER_SIZE,
        batch_size  = BATCH_SIZE,
    )

    if verbose:
        print("=" * 62)
        print("DQN 训练启动")
        print(f"  状态维度:     {env.state_dim}")
        print(f"  动作维度:     {env.action_dim}")
        print(f"  训练 episode: {n_episodes}")
        print(f"  PoI 子集大小: {env.P}")
        print(f"  资源预算:     {env.K_total} units (UAV=2, Participant=1)")
        print("=" * 62)

    grqi_log    = []
    reward_log  = []
    loss_log    = []
    t_start     = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done  = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            valid_mask = env.get_valid_mask()

            # ε-greedy 选动作
            action = agent.act(state, valid_mask)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存入回放缓冲区
            agent.memory.push(state, action, reward, next_state, done)

            # 更新 Q 网络
            loss = agent.train()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward

        agent.increment_episode()

        # 同步目标网络
        if ep % TARGET_SYNC_EVERY == 0:
            agent.sync_target()

        grqi_log.append(env.current_grqi)
        reward_log.append(ep_reward)
        if ep_losses:
            loss_log.append(np.mean(ep_losses))

        # 日志
        if verbose and ep % LOG_EVERY == 0:
            avg_grqi   = np.mean(grqi_log[-LOG_EVERY:])
            avg_reward = np.mean(reward_log[-LOG_EVERY:])
            avg_loss   = np.mean(loss_log[-min(LOG_EVERY, len(loss_log)):]) if loss_log else 0.0
            elapsed    = time.time() - t_start
            eta        = elapsed / ep * (n_episodes - ep)
            print(
                f"  Ep {ep:4d}/{n_episodes}  "
                f"GRQI={avg_grqi:.4f}  "
                f"Reward={avg_reward:.4f}  "
                f"Loss={avg_loss:.5f}  "
                f"ε={agent.epsilon:.3f}  "
                f"Elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    if verbose:
        elapsed = time.time() - t_start
        print(f"\n训练完成！耗时 {elapsed:.1f}s")
        print(f"最终 GRQI（最近 50 ep 均值）: {np.mean(grqi_log[-50:]):.4f}")

    # 保存模型
    agent.save(RL_SAVE_PATH)
    return agent, grqi_log, reward_log


# ─────────────────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_rl(env: MCSEnv, agent: DQNAgent) -> Dict:
    """运行一个 episode（无探索，纯贪心）"""
    agent.q_net.eval()
    state = env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        valid_mask = env.get_valid_mask()
        # 评估时 ε=0（完全贪心）
        state_t = state.unsqueeze(0)
        with torch.no_grad():
            q_vals = agent.q_net(state_t).squeeze(0)
        q_vals[~valid_mask] = -1e9
        action = int(q_vals.argmax().item())

        state, reward, done, info = env.step(action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}


def evaluate_all(env: MCSEnv, agent: DQNAgent,
                 n_eval: int = N_EVAL_EPISODES) -> Dict[str, Dict]:
    """
    对 5 种方法各运行 n_eval 个 episode，计算统计量。

    Returns:
        {method_name: {'grqi_mean', 'grqi_std', 'grqi_max',
                       'n_uav_mean', 'n_enhanced_mean'}}
    """
    methods = {
        'RL (DQN)':         lambda: run_episode_rl(env, agent),
        'Random':           lambda: run_episode_random(env),
        'UAV-Only-Random':  lambda: run_episode_uav_only(env),
        'Greedy-Uncert':    lambda: run_episode_greedy_uncertainty(env),
        'Greedy-GRQI':      lambda: run_episode_greedy_grqi(env),
    }

    results = {}
    for name, run_fn in methods.items():
        grqis      = []
        rewards    = []
        n_uavs     = []
        n_enhanced = []

        for _ in range(n_eval):
            info = run_fn()
            grqis.append(info['grqi'])
            rewards.append(info['reward'])
            n_uavs.append(info['n_uav'])
            n_enhanced.append(info['n_enhanced'])

        results[name] = {
            'grqi_mean':      float(np.mean(grqis)),
            'grqi_std':       float(np.std(grqis)),
            'grqi_max':       float(np.max(grqis)),
            'n_uav_mean':     float(np.mean(n_uavs)),
            'n_enh_mean':     float(np.mean(n_enhanced)),
            'reward_mean':    float(np.mean(rewards)),
        }

    return results


def print_results(results: Dict[str, Dict], baseline_grqi: float):
    """打印最终对比结果表"""
    print()
    print("=" * 74)
    print("最终对比结果（GRQI：越高越好，范围 (-inf, 1.0]）")
    print(f"零分配基线 GRQI（无任何 UAV/增强）: {baseline_grqi:.4f}")
    print("=" * 74)
    print(f"{'方法':<18} {'GRQI均值':>9} {'GRQI标准差':>10} {'GRQI最高':>9} "
          f"{'UAV平均':>8} {'增强平均':>8} {'vs基线提升':>10}")
    print("-" * 74)

    for name, r in results.items():
        delta = r['grqi_mean'] - baseline_grqi
        prefix = ">> " if name == 'RL (DQN)' else "   "
        print(f"{prefix}{name:<15} {r['grqi_mean']:>9.4f} {r['grqi_std']:>10.4f} "
              f"{r['grqi_max']:>9.4f} {r['n_uav_mean']:>8.1f} "
              f"{r['n_enh_mean']:>8.1f} {delta:>+10.4f}")

    print("=" * 74)
    rl_grqi = results.get('RL (DQN)', {}).get('grqi_mean', 0)
    rand_grqi = results.get('Random', {}).get('grqi_mean', 0)
    if rand_grqi > 0:
        rl_vs_rand = (rl_grqi - rand_grqi) / abs(rand_grqi) * 100
        print(f"RL vs Random: {rl_vs_rand:+.1f}%")
    grqi_base = results.get('Greedy-GRQI', {}).get('grqi_mean', 0)
    if grqi_base > 0:
        rl_vs_greedy = (rl_grqi - grqi_base) / abs(grqi_base) * 100
        print(f"RL vs Greedy-GRQI: {rl_vs_greedy:+.1f}%")
    print()
