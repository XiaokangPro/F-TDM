#!/usr/bin/env python3
# rl_extension/run_ppo.py
# PPO vs DQN 对比实验
#
# 目标：验证 PPO 能否突破 DQN 的 ~0.81 天花板，冲击 0.87+ GRQI
#
# 训练规模（公平对比，均约 3000 episode 等效）：
#   DQN Baseline : 3000 episodes，ε-greedy 探索
#   PPO          : 150 次外迭代 × 每次收集 20 episodes = 3000 episodes
#
# 运行：
#   cd ftdm_project
#   python rl_extension/run_ppo.py

import os, sys, random, time
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
from rl_extension.dqn_agent import DQNAgent
from rl_extension.ppo_agent import PPOAgent
from rl_extension.rl_train import LR, GAMMA, BATCH_SIZE, TARGET_SYNC_EVERY

# ─────────────────────────────────────────────────────────────────────────────
# 超参
# ─────────────────────────────────────────────────────────────────────────────

# PPO
PPO_HIDDEN        = 256
PPO_LR            = 3e-4
PPO_CLIP          = 0.2
PPO_EPOCHS        = 4
PPO_GAE_LAMBDA    = 0.95
PPO_VF_COEF       = 0.5
PPO_ENT_COEF      = 0.002  # 低熵系数，让策略收敛而非维持均匀分布
PPO_MAX_GRAD      = 0.5
PPO_ROLLOUT_EP    = 20     # 每次外迭代收集的 episode 数
PPO_OUTER_ITER    = 200    # 外迭代次数（200 × 20 = 4000 episodes）

# DQN（对照）
DQN_EPISODES      = 4000
DQN_EPS_DECAY     = 2000
DQN_BUFFER        = 10000

# 评估
N_EVAL            = 50
LOG_EVERY_ITER    = 15    # PPO 每多少次外迭代打印一次（≈每300ep）
LOG_EVERY_DQN     = 300   # DQN 每多少episode打印一次


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────────────────────
# PPO 训练
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(env, n_outer_iter, n_rollout_ep, verbose=True):
    """
    PPO 主训练循环。

    每次外迭代：
      1. 用当前策略收集 n_rollout_ep 个完整 episode
      2. 用全部收集数据做 PPO_EPOCHS 轮梯度更新
      3. 清空缓冲区，重复

    Returns: (agent, grqi_log_per_episode)
    """
    agent    = PPOAgent(
        state_dim   = env.state_dim,
        action_dim  = env.action_dim,
        hidden      = PPO_HIDDEN,
        lr          = PPO_LR,
        clip_eps    = PPO_CLIP,
        n_epochs    = PPO_EPOCHS,
        gamma       = GAMMA,
        gae_lambda  = PPO_GAE_LAMBDA,
        vf_coef     = PPO_VF_COEF,
        ent_coef    = PPO_ENT_COEF,
        max_grad    = PPO_MAX_GRAD,
    )

    if verbose:
        print(f"  PPO 网络参数量: {agent.count_params():,}")

    grqi_log  = []   # 每个 episode 的最终 GRQI
    t0        = time.time()

    for outer in range(1, n_outer_iter + 1):

        # ── 收集轨迹 ────────────────────────────────────────────────────
        ep_grqis = []
        for _ in range(n_rollout_ep):
            state = env.reset()
            done  = False
            while not done:
                mask              = env.get_valid_mask()
                action, lp, val   = agent.get_action(state, mask)
                next_state, r, done, _ = env.step(action)
                agent.buffer.add(state, action, lp, r, val, done, mask)
                state = next_state
            ep_grqis.append(env.current_grqi)

        grqi_log.extend(ep_grqis)

        # ── PPO 更新 ────────────────────────────────────────────────────
        loss_info = agent.update()

        # ── 日志 ────────────────────────────────────────────────────────
        if verbose and outer % LOG_EVERY_ITER == 0:
            ep_num = outer * n_rollout_ep
            avg    = np.mean(grqi_log[-LOG_EVERY_ITER * n_rollout_ep:])
            print(f"    [PPO] iter {outer:4d}/{n_outer_iter}  "
                  f"ep={ep_num:5d}  avg_GRQI={avg:.4f}  "
                  f"actor={loss_info['actor']:.4f}  "
                  f"ent={loss_info['entropy']:.4f}  "
                  f"t={time.time()-t0:.0f}s")

    return agent, np.array(grqi_log)


# ─────────────────────────────────────────────────────────────────────────────
# DQN 训练（对照）
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn_baseline(env, n_episodes, verbose=True):
    agent = DQNAgent(
        state_dim   = env.state_dim,
        action_dim  = env.action_dim,
        lr          = LR,
        gamma       = GAMMA,
        eps_decay   = DQN_EPS_DECAY,
        buffer_size = DQN_BUFFER,
        batch_size  = BATCH_SIZE,
    )
    grqi_log = []
    t0       = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset(); done = False
        while not done:
            mask   = env.get_valid_mask()
            action = agent.act(state, mask)
            ns, r, done, _ = env.step(action)
            agent.memory.push(state, action, r, ns, done)
            agent.train()
            state = ns
        agent.increment_episode()
        if ep % TARGET_SYNC_EVERY == 0:
            agent.sync_target()
        grqi_log.append(env.current_grqi)
        if verbose and ep % LOG_EVERY_DQN == 0:
            avg = np.mean(grqi_log[-LOG_EVERY_DQN:])
            print(f"    [DQN] Ep {ep:5d}/{n_episodes}  "
                  f"GRQI={avg:.4f}  ε={agent.epsilon:.3f}  t={time.time()-t0:.0f}s")

    return agent, np.array(grqi_log)


# ─────────────────────────────────────────────────────────────────────────────
# 贪心评估（PPO）
# ─────────────────────────────────────────────────────────────────────────────

def eval_ppo(env, agent: PPOAgent, n_eval: int) -> np.ndarray:
    agent.policy.eval()
    grqis = []
    for _ in range(n_eval):
        state = env.reset(); done = False
        while not done:
            mask = env.get_valid_mask()
            with torch.no_grad():
                logits, _ = agent.policy(state.unsqueeze(0), mask.unsqueeze(0))
            action = int(logits.squeeze(0).argmax().item())
            state, _, done, _ = env.step(action)
        grqis.append(env.current_grqi)
    return np.array(grqis)


def eval_dqn(env, agent: DQNAgent, n_eval: int) -> np.ndarray:
    agent.q_net.eval()
    grqis = []
    for _ in range(n_eval):
        state = env.reset(); done = False
        while not done:
            mask = env.get_valid_mask()
            with torch.no_grad():
                q = agent.q_net(state.unsqueeze(0)).squeeze(0)
            q[~mask] = -1e9
            state, _, done, _ = env.step(int(q.argmax()))
        grqis.append(env.current_grqi)
    return np.array(grqis)


def sample_eff(log, thr, w=30):
    for i in range(w - 1, len(log)):
        if np.mean(log[i - w + 1:i + 1]) >= thr:
            return i + 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    os.makedirs('checkpoints', exist_ok=True)

    # ── 加载 F-TDM + 数据 ─────────────────────────────────────────────
    print("[1] 加载 F-TDM 和 NO_x 测试数据...")
    if not os.path.exists(Config.SAVE_PATH):
        print(f"  错误：请先运行 main.py 完成 F-TDM 元训练")
        sys.exit(1)
    model = TruthDiscoveryNet(Config.INPUT_SIZE, Config.HIDDEN_SIZE)
    ftdm  = FTDM(model); ftdm.load(Config.SAVE_PATH)
    df = load_air_quality('data/AirQualityUCI.csv')
    _, D_W_test, D_U_test, _ = build_all_data(df)
    print(f"  NO_x 测试集: {D_W_test.shape[0]} PoI")

    # 环境工厂（纯 GRQI 奖励，随机子集）
    def make_env():
        return MCSEnv(
            ftdm=ftdm, D_W_pool=D_W_test, D_U_pool=D_U_test,
            p_size=P_SIZE, k_total=K_TOTAL, k_finetune=1,
            use_anchor=True, alpha=1.0, beta=0.0, gamma=0.0,
            fixed_subset=False,
        )

    total_ppo_ep = PPO_OUTER_ITER * PPO_ROLLOUT_EP
    print(f"\n  DQN 训练规模: {DQN_EPISODES} episodes")
    print(f"  PPO 训练规模: {PPO_OUTER_ITER} iters × {PPO_ROLLOUT_EP} ep = {total_ppo_ep} episodes")

    # ── DQN 对照 ─────────────────────────────────────────────────────
    print(f"\n[2] DQN 训练（{DQN_EPISODES} episodes）...")
    env_dqn = make_env()
    set_seed(42)
    dqn_agent, log_dqn = train_dqn_baseline(env_dqn, DQN_EPISODES)
    env_dqn.k_ft = 5
    set_seed(0); eval_dqn_res = eval_dqn(env_dqn, dqn_agent, N_EVAL)
    print(f"  DQN 评估 GRQI = {eval_dqn_res.mean():.4f} ± {eval_dqn_res.std():.4f}")

    # ── PPO 训练 ─────────────────────────────────────────────────────
    print(f"\n[3] PPO 训练（{PPO_OUTER_ITER} 次外迭代，每次 {PPO_ROLLOUT_EP} episodes）...")
    env_ppo = make_env()
    set_seed(42)
    ppo_agent, log_ppo = train_ppo(env_ppo, PPO_OUTER_ITER, PPO_ROLLOUT_EP)
    env_ppo.k_ft = 5
    set_seed(0); eval_ppo_res = eval_ppo(env_ppo, ppo_agent, N_EVAL)
    print(f"  PPO 评估 GRQI = {eval_ppo_res.mean():.4f} ± {eval_ppo_res.std():.4f}")

    # ── 结果汇总 ──────────────────────────────────────────────────────
    THRESHOLDS = [0.83, 0.85, 0.87, 0.89]

    print()
    print("=" * 70)
    print(f"最终对比（专家 Greedy-GRQI ≈ 0.91，零分配基线 ≈ 0.06）")
    print("=" * 70)

    rows = [
        ('DQN Baseline',  eval_dqn_res, log_dqn),
        ('PPO',           eval_ppo_res, log_ppo),
    ]

    print(f"\n{'方法':16} {'GRQI均值':>9} {'标准差':>8} {'最高':>8} {'vs DQN':>8}")
    print("-" * 55)
    dqn_mean = eval_dqn_res.mean()
    for name, ev, _ in rows:
        vs = ev.mean() - dqn_mean
        best = ' <<最优' if ev.mean() == max(r[1].mean() for r in rows) else ''
        print(f"  {name:14} {ev.mean():>9.4f} {ev.std():>8.4f} "
              f"{ev.max():>8.4f} {vs:>+8.4f}{best}")

    print(f"\n样本效率（首次达到 GRQI 阈值的 episode，滑动窗口=30）：")
    print(f"  {'方法':16}" + "".join(f" {t:>8.2f}" for t in THRESHOLDS))
    print("-" * 55)
    for name, _, log in rows:
        row = f"  {name:16}"
        for t in THRESHOLDS:
            ep = sample_eff(log, t)
            row += f" {'未达到':>8}" if ep is None else f" {ep:>8}"
        print(row)

    # 分段训练曲线（每 300 episode 均值）
    seg = 300
    print(f"\n训练收敛曲线（每 {seg} episode 均值）：")
    print(f"  {'Episode':>10}  {'DQN':>10}  {'PPO':>10}  {'差值':>10}")
    print("-" * 48)
    n_segs = min(len(log_dqn), len(log_ppo)) // seg
    for k in range(n_segs):
        ep  = (k + 1) * seg
        d   = float(np.mean(log_dqn[k*seg:(k+1)*seg]))
        p   = float(np.mean(log_ppo[k*seg:(k+1)*seg]))
        mark = ' <<' if abs(p - d) > 0.01 else ''
        print(f"  {ep:>10}  {d:>10.4f}  {p:>10.4f}  {p-d:>+10.4f}{mark}")

    print()
    print("=" * 70)
    ppo_mean = eval_ppo_res.mean()
    print("核心结论：")
    if ppo_mean >= 0.87:
        print(f"  PPO 达到 0.87+ 目标！GRQI = {ppo_mean:.4f}")
        print(f"  比 DQN 提升 {ppo_mean - dqn_mean:+.4f}，PPO 算法优势已充分验证。")
    elif ppo_mean >= 0.85:
        print(f"  PPO 达到 0.85+，GRQI = {ppo_mean:.4f}，比 DQN 提升 {ppo_mean-dqn_mean:+.4f}")
        print("  可通过增加外迭代次数（PPO_OUTER_ITER）进一步接近 0.87+。")
    elif ppo_mean > dqn_mean + 0.005:
        print(f"  PPO 显著优于 DQN：{ppo_mean:.4f} vs {dqn_mean:.4f} (+{ppo_mean-dqn_mean:.4f})")
        print("  继续增加训练量有望突破 0.87+。")
    else:
        print(f"  PPO ({ppo_mean:.4f}) vs DQN ({dqn_mean:.4f})，差异 {ppo_mean-dqn_mean:+.4f}")
        print("  PPO 在此规模下优势未完全体现，可调整 PPO_ENT_COEF 或增加训练量。")
    print("=" * 70)


if __name__ == '__main__':
    main()
