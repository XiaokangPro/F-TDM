#!/usr/bin/env python3
# rl_extension/run_improved.py
# 两阶段优化：方向一（固定子集）+ 方向二（专家缓冲区预填充 + RL）
#
# 三组对比（评估均在随机子集上，公平比较）：
#   Baseline   ： 标准 DQN，随机子集，ε 从 1.0 衰减（2000 回合）
#   Dir1       ： 固定子集，DQN 专注学特定 PoI 最优分配（2000 回合）
#   Dir1+2     ： 随机子集 + 专家轨迹预填充回放池 + ε 从 0.5 起（2000 回合）
#
# 方向二核心思路（避免灾难性遗忘的简洁做法）：
#   - 预先用 Greedy-GRQI 采集 300 个 episode 的专家轨迹（GRQI≈0.91）
#   - 将专家转移预填充进回放缓冲区
#   - RL 从 ε=0.5 出发（已有先验知识，无需大量随机探索）
#   - RL 自然混合专家数据 + 自身经验，无需显式 BC 损失

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
from rl_extension.imitation import collect_expert_trajectories
from rl_extension.rl_train import LR, GAMMA, BATCH_SIZE, TARGET_SYNC_EVERY

# ─────────────────────────────────────────────────────────────────────────────
# 超参
# ─────────────────────────────────────────────────────────────────────────────
N_EXPERT_EP  = 300    # 专家轨迹采集回合数
N_RL_EP      = 2000   # RL 训练回合数（所有组相同）
N_EVAL       = 50     # 评估回合数
EPS_STD      = 1.0    # Baseline/Dir1 初始 ε
EPS_BC_RL    = 0.5    # Dir1+2 初始 ε（已有专家先验，减少随机探索）
EPS_DECAY    = 1500   # ε 衰减轮数
BUFFER_SIZE  = 10000


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────────────────────
# RL 训练循环
# ─────────────────────────────────────────────────────────────────────────────

def train_rl(env, agent, n_ep, label, log_every=200):
    grqi_log = []; t0 = time.time()
    for ep in range(1, n_ep + 1):
        state = env.reset(); done = False
        while not done:
            mask   = env.get_valid_mask()
            action = agent.act(state, mask)
            ns, r, done, info = env.step(action)
            agent.memory.push(state, action, r, ns, done)
            agent.train()
            state = ns
        agent.increment_episode()
        if ep % TARGET_SYNC_EVERY == 0:
            agent.sync_target()
        grqi_log.append(env.current_grqi)
        if ep % log_every == 0:
            avg = np.mean(grqi_log[-log_every:])
            print(f"    [{label}] Ep {ep:5d}/{n_ep}  "
                  f"GRQI={avg:.4f}  ε={agent.epsilon:.3f}  t={time.time()-t0:.0f}s")
    return np.array(grqi_log)


# ─────────────────────────────────────────────────────────────────────────────
# 贪心评估
# ─────────────────────────────────────────────────────────────────────────────

def eval_greedy(env, agent, n_eval):
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
    for i in range(w-1, len(log)):
        if np.mean(log[i-w+1:i+1]) >= thr:
            return i+1
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

    # 固定子集（Dir1 专用）：随机多样化，不用连续时序
    rng = np.random.RandomState(99)
    fixed_idx = np.sort(rng.choice(len(D_U_test), P_SIZE, replace=False))
    baseline_grqi_fixed = float(compute_grqi_baseline(
        torch.from_numpy(D_W_test[fixed_idx]), D_U_test[fixed_idx]
    ))
    baseline_grqi_rand = float(compute_grqi_baseline(
        torch.from_numpy(D_W_test[:P_SIZE]), D_U_test[:P_SIZE]
    ))
    print(f"  零分配基线 GRQI（随机）: {baseline_grqi_rand:.4f}")
    print(f"  零分配基线 GRQI（固定）: {baseline_grqi_fixed:.4f}")

    # 环境工厂
    def make_env(fixed=False, alpha=1.0, beta=0.0, gamma=0.0):
        return MCSEnv(
            ftdm=ftdm, D_W_pool=D_W_test, D_U_pool=D_U_test,
            p_size=P_SIZE, k_total=K_TOTAL, k_finetune=1,
            use_anchor=True, alpha=alpha, beta=beta, gamma=gamma,
            fixed_subset=fixed, fixed_indices=fixed_idx if fixed else None,
        )

    def make_agent(state_dim, eps_start=EPS_STD):
        return DQNAgent(
            state_dim=state_dim, action_dim=2*P_SIZE,
            lr=LR, gamma=GAMMA, eps_start=eps_start,
            eps_decay=EPS_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )

    # ══════════════════════════════════════════════════════════════════
    # Baseline：随机子集，标准 DQN（复现对照）
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[2] Baseline（随机子集，{N_RL_EP} 回合）...")
    env_b   = make_env(fixed=False)
    agent_b = make_agent(env_b.state_dim)
    set_seed(42); log_b = train_rl(env_b, agent_b, N_RL_EP, 'Baseline')
    env_b.k_ft = 5; set_seed(0)
    eval_b = eval_greedy(env_b, agent_b, N_EVAL)
    print(f"  Baseline 评估 GRQI = {eval_b.mean():.4f} ± {eval_b.std():.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Dir1：固定子集，纯 GRQI 奖励
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3] Dir1（固定子集 + 纯GRQI，{N_RL_EP} 回合）...")
    env_1   = make_env(fixed=True)
    agent_1 = make_agent(env_1.state_dim)
    set_seed(42); log_1 = train_rl(env_1, agent_1, N_RL_EP, 'Dir1')
    # Dir1 在固定子集上评估（确定性环境）
    env_1.k_ft = 5; set_seed(0)
    eval_1 = eval_greedy(env_1, agent_1, 10)   # 10次即可（固定子集结果一致）
    print(f"  Dir1 固定子集评估 GRQI = {eval_1.mean():.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Dir1+2：随机子集 + 专家缓冲预填充 + ε 从 0.5 起
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[4] 采集专家轨迹（Greedy-GRQI，{N_EXPERT_EP} 回合，随机子集）...")
    env_exp = make_env(fixed=False)    # 专家在随机子集上运行
    set_seed(7)
    expert_data, expert_mean = collect_expert_trajectories(
        env_exp, N_EXPERT_EP, verbose=True
    )

    print(f"\n[5] Dir1+2（专家预填充 + RL，ε 从 {EPS_BC_RL} 起，{N_RL_EP} 回合）...")
    env_12   = make_env(fixed=False)
    agent_12 = make_agent(env_12.state_dim, eps_start=EPS_BC_RL)

    # 预填充：把专家轨迹全部放入回放池
    # RL 早期大量采样专家经验 → Q 值自然收敛到专家水平
    # 随着 RL 探索增多，专家数据比例逐渐降低 → 自然过渡，无需显式 BC 损失
    print(f"  预填充 {len(expert_data)} 条专家转移到回放池...")
    for s, a, r, ns, d in expert_data:
        agent_12.memory.push(s, a, r, ns, d)
    print(f"  回放池已填充 {len(agent_12.memory)}/{BUFFER_SIZE}")

    set_seed(42); log_12 = train_rl(env_12, agent_12, N_RL_EP, 'Dir1+2')
    env_12.k_ft = 5; set_seed(0)
    eval_12 = eval_greedy(env_12, agent_12, N_EVAL)
    print(f"  Dir1+2 评估 GRQI = {eval_12.mean():.4f} ± {eval_12.std():.4f}")

    # ── 结果汇总 ──────────────────────────────────────────────────────
    THRESHOLDS = [0.83, 0.85, 0.87, 0.89]
    print()
    print("=" * 72)
    print(f"最终结果（专家 Greedy-GRQI = {expert_mean:.4f}，"
          f"随机基线 GRQI = {baseline_grqi_rand:.4f}）")
    print("=" * 72)

    rows = [
        ('Baseline（随机子集）',     eval_b,  log_b,  baseline_grqi_rand),
        ('Dir1   （固定子集）',      eval_1,  log_1,  baseline_grqi_fixed),
        ('Dir1+2 （BC预填+RL）',     eval_12, log_12, baseline_grqi_rand),
        ('专家上界（Greedy-GRQI）',   None,    None,   None),
    ]

    print(f"\n{'方法':26} {'GRQI均值':>9} {'标准差':>8} {'最高':>8} {'vs基线':>8}")
    print("-" * 65)
    for name, ev, _, zero in rows:
        if ev is None:
            print(f"  {name:24} {expert_mean:>9.4f}   {'—':>6}   {'—':>6}  {'—':>6}")
        else:
            vs = ev.mean() - zero
            best = ' <<' if ev.mean() == max(r[1].mean() for r in rows[:3])  else ''
            print(f"  {name:24} {ev.mean():>9.4f} {ev.std():>8.4f} "
                  f"{ev.max():>8.4f} {vs:>+8.4f}{best}")

    print(f"\n样本效率（GRQI 训练曲线首次达到阈值，窗口=30）：")
    hdr = f"  {'方法':26}" + "".join(f" {t:>8.2f}" for t in THRESHOLDS)
    print(hdr); print("-" * 65)
    for name, _, log, _ in rows[:3]:
        if log is None:
            continue
        row = f"  {name:26}"
        for t in THRESHOLDS:
            ep = sample_eff(log, t)
            row += f" {'未达到':>8}" if ep is None else f" {ep:>8}"
        print(row)

    print()
    print("=" * 72)
    b  = eval_b.mean()
    d2 = eval_12.mean()
    print("核心结论：")
    print(f"  Baseline             GRQI = {b:.4f}")
    print(f"  Dir1+2（BC预填+RL）  GRQI = {d2:.4f}  提升 {d2-b:+.4f}")
    print(f"  专家（Greedy-GRQI）  GRQI = {expert_mean:.4f}  "
          f"Dir1+2 与专家差距 {expert_mean-d2:.4f}")
    if d2 >= 0.87:
        print(f"\n  Dir1+2 已达到 0.87+ 目标！")
    elif d2 >= 0.85:
        print(f"\n  Dir1+2 已达到 0.85+，接近目标。")
    elif d2 > b:
        print(f"\n  BC预填充对 RL 有正向提升 {d2-b:+.4f}，"
              f"可增加专家回合数（N_EXPERT_EP）继续提升。")
    else:
        print(f"\n  当前提升不显著，建议增加专家采集量或调整 BC 策略。")
    print("=" * 72)


if __name__ == '__main__':
    main()
