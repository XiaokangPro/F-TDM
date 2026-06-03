# rl_extension/imitation.py
# 模仿学习模块（Imitation Learning / Behavioral Cloning）
#
# 原理：
#   Greedy-GRQI 每步穷举所有动作计算真实 GRQI 增益，选最优者。
#   这是当前实现下的"专家策略"（GRQI = 0.91）。
#
#   行为克隆（BC）：用专家轨迹预训练 DQN，使其 Q 网络输出
#   在专家动作上的值显著高于其他动作（Large Margin Loss）。
#
#   BC 预训练 → RL 微调（DQfD 思路）：
#     从好的初始化出发，RL 只需小范围微调即可超越专家。

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

from rl_extension.grqi import compute_grqi
from rl_extension.mcs_env import LAMBDA_ENH, COST_UAV, COST_PART


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 1：专家轨迹采集（Greedy-GRQI 作为专家策略）
# ─────────────────────────────────────────────────────────────────────────────

def collect_expert_trajectories(env, n_episodes: int,
                                verbose: bool = True) -> List[tuple]:
    """
    用 Greedy-GRQI 在环境上跑 n_episodes 个 episode，收集专家轨迹。

    专家策略：每步枚举所有合法动作，直接计算 GRQI 增益，选最大者。
    （无需 env.step 回滚，直接用 compute_grqi 模拟）

    Returns:
        expert_data: list of (state, action, reward, next_state, done)
    """
    expert_data = []
    grqi_log    = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done  = False

        while not done:
            mask  = env.get_valid_mask()
            valid = mask.nonzero(as_tuple=True)[0].tolist()
            if not valid:
                break

            P           = env.P
            cur_grqi    = env.current_grqi
            cur_uav     = list(env._uav_visited)
            cur_dw_eff  = env._D_W_eff

            # ── 穷举选最优动作（Greedy-GRQI 逻辑）────────────────────
            best_action = valid[0]
            best_delta  = -float('inf')

            for action in valid:
                temp_uav    = list(cur_uav)
                temp_dw_eff = cur_dw_eff.copy()

                if action < P:
                    temp_uav.append(action)
                else:
                    poi_i = action - P
                    row_mean = temp_dw_eff[poi_i].mean()
                    temp_dw_eff[poi_i] = (LAMBDA_ENH * row_mean
                                          + (1 - LAMBDA_ENH) * temp_dw_eff[poi_i])

                new_grqi = compute_grqi(
                    env.ftdm,
                    torch.from_numpy(temp_dw_eff.astype('float32')),
                    env._D_U,
                    temp_uav,
                    k_finetune=1,
                )
                delta = new_grqi - cur_grqi
                if delta > best_delta:
                    best_delta  = delta
                    best_action = action

            # ── 执行专家动作，记录轨迹 ──────────────────────────────────
            next_state, reward, done, info = env.step(best_action)
            expert_data.append((state, best_action, reward, next_state, done))
            state = next_state

        grqi_log.append(env.current_grqi)

        if verbose and ep % max(1, n_episodes // 5) == 0:
            print(f"    专家采集 {ep:4d}/{n_episodes}  "
                  f"avg_GRQI={np.mean(grqi_log[-max(1,n_episodes//5):]):.4f}")

    expert_mean = float(np.mean(grqi_log))
    print(f"  专家策略平均 GRQI: {expert_mean:.4f}  （{len(expert_data)} 条轨迹）")
    return expert_data, expert_mean


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 2：行为克隆预训练
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_bc(agent, expert_data: List[tuple],
                n_steps: int = 2000,
                margin:  float = 0.8,
                lam_bc:  float = 1.0,
                lam_q:   float = 0.5,
                batch_size: int = 64,
                verbose: bool = True) -> None:
    """
    Large Margin 行为克隆预训练。

    损失函数（DQfD Large Margin Loss）：
      L = λ_q · L_Q  +  λ_bc · L_BC

      L_Q  = 标准 Q-learning 损失（让 Q 值估计准确）
      L_BC = max(0, max_{a ≠ a_E}[Q(s,a)] + margin − Q(s, a_E))
             （让专家动作的 Q 值比所有其他动作高出 margin）

    参数：
      margin:  Q(s, a_E) 相对于最优非专家动作的最小优势（越大越保守）
      lam_bc:  BC loss 权重
      lam_q:   Q-learning loss 权重

    副作用：直接修改 agent.q_net 的权重（无返回值）
    """
    # 先把专家数据填入回放缓冲区（供标准 Q-learning 使用）
    for s, a, r, ns, d in expert_data:
        agent.memory.push(s, a, r, ns, d)

    agent.q_net.train()

    for step in range(1, n_steps + 1):
        if len(agent.memory) < batch_size:
            continue

        # ── 随机采样 batch ─────────────────────────────────────────────
        batch = random.sample(expert_data, min(batch_size, len(expert_data)))
        states  = torch.stack([b[0] for b in batch])           # (B, state_dim)
        actions = torch.tensor([b[1] for b in batch])          # (B,)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones   = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_vals = agent.q_net(states)                           # (B, action_dim)

        # ── L_BC：Large Margin Loss ───────────────────────────────────
        q_expert = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)

        # 对每个样本屏蔽专家动作，找最大非专家 Q 值
        q_masked = q_vals.clone()
        q_masked.scatter_(1, actions.unsqueeze(1), -1e9)
        q_best_other = q_masked.max(dim=1).values                      # (B,)

        l_bc = F.relu(q_best_other + margin - q_expert).mean()

        # ── L_Q：标准 Q-learning ──────────────────────────────────────
        with torch.no_grad():
            next_q = agent.target_net(next_states).max(dim=1).values
            q_target = rewards + agent.gamma * next_q * (1 - dones)

        l_q = F.smooth_l1_loss(q_expert, q_target)

        # ── 联合损失 ──────────────────────────────────────────────────
        loss = lam_q * l_q + lam_bc * l_bc

        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), 1.0)
        agent.optimizer.step()

        if verbose and step % (n_steps // 5) == 0:
            print(f"    BC 预训练 {step:5d}/{n_steps}  "
                  f"loss={loss.item():.5f}  "
                  f"l_bc={l_bc.item():.5f}  l_q={l_q.item():.5f}")

    # 同步目标网络
    agent.sync_target()
    print("  BC 预训练完成，目标网络已同步。")
