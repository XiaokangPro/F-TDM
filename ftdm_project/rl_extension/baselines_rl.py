# rl_extension/baselines_rl.py
# RL 对比基线：4 种确定性/启发式分配策略
#
# 所有基线在相同 MCSEnv 上运行，公平对比 episode 结束时的 GRQI
#
# 基线列表：
#   1. Random           — 每步随机选一个合法动作
#   2. UAV-Only-Random  — 只使用 UAV 动作，随机选 PoI（不做参与者增强）
#   3. Greedy-Uncert    — 优先给人类分歧（D_W std）最大的 PoI 分配 UAV
#   4. Greedy-GRQI      — 每步枚举所有合法动作，选令 GRQI 提升最大的（计算密集但接近最优）

import random
import numpy as np
import torch
from typing import Dict


def run_episode_random(env) -> Dict:
    """
    随机分配基线：每步从合法动作中均匀随机选择。
    不区分 UAV 和参与者增强，完全无先验知识。
    """
    state = env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        mask  = env.get_valid_mask()
        valid = mask.nonzero(as_tuple=True)[0].tolist()
        if not valid:
            break
        action = random.choice(valid)
        state, reward, done, info = env.step(action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}


def run_episode_uav_only(env) -> Dict:
    """
    仅 UAV 分配基线：随机选 PoI 发送 UAV，不做参与者增强。
    模拟只使用 UAV 但不优化选点的策略（UAV 预算有限）。
    """
    P = env.P
    state = env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        mask  = env.get_valid_mask()
        # 只考虑 UAV 动作 [0, P)
        uav_mask = mask.clone()
        uav_mask[P:] = False    # 屏蔽所有参与者增强动作

        valid = uav_mask.nonzero(as_tuple=True)[0].tolist()
        if not valid:
            # UAV 预算耗尽，剩余预算用于参与者增强（随机）
            part_valid = mask.nonzero(as_tuple=True)[0].tolist()
            if not part_valid:
                break
            action = random.choice(part_valid)
        else:
            action = random.choice(valid)

        state, reward, done, info = env.step(action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}


def run_episode_greedy_uncertainty(env) -> Dict:
    """
    不确定性贪心基线：优先给人类数据分歧（D_W 行标准差）最大的 PoI 分配 UAV。

    直觉：高不确定性 PoI 的人类数据质量差，UAV 标定收益最大。
    这是常见的主动学习（Active Learning）策略。
    """
    from rl_extension.anchor import compute_anchor_scores

    P     = env.P
    state = env.reset()
    done  = False
    total_reward = 0.0

    # 用综合锚点分（方差+冲突度+偏离均值）替代原始标准差
    # 对应论文创新点一："量化分析多维感知数据的方差与冲突度"
    anchor_scores = compute_anchor_scores(env._D_W)   # (P,)，越高越应优先 UAV

    while not done:
        mask  = env.get_valid_mask()

        # 优先选 UAV 动作：锚点分最高的 PoI 最需要 UAV 校验
        uav_valid = [i for i in range(P) if mask[i]]
        if uav_valid:
            action = max(uav_valid, key=lambda i: anchor_scores[i])
        else:
            # UAV 预算耗尽，转用参与者增强，同样选锚点分最高的
            part_valid = [i - P for i in range(P, 2*P) if mask[i]]
            if not part_valid:
                break
            action = max(part_valid, key=lambda i: anchor_scores[i]) + P

        state, reward, done, info = env.step(action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}


def run_episode_greedy_grqi(env) -> Dict:
    """
    GRQI 贪心基线（接近最优上界）：
    每步枚举所有合法动作，直接计算每种选择后的 GRQI 提升，选最优动作。

    不使用 env.step 做模拟（避免快照问题），而是直接调用 compute_grqi。
    计算量大，是性能上界参照。
    """
    import torch as _torch
    from rl_extension.grqi import compute_grqi as _compute_grqi
    from rl_extension.mcs_env import LAMBDA_ENH, COST_UAV, COST_PART

    state = env.reset()
    done  = False
    total_reward = 0.0
    info  = {'n_uav': 0, 'n_enhanced': 0, 'budget_left': env.K_total}

    while not done:
        mask  = env.get_valid_mask()
        valid = mask.nonzero(as_tuple=True)[0].tolist()
        if not valid:
            break

        P            = env.P
        cur_grqi     = env.current_grqi
        cur_uav      = list(env._uav_visited)
        cur_dw_eff   = env._D_W_eff          # (P, I) numpy

        best_action = valid[0]
        best_delta  = -float('inf')

        for action in valid:
            # 模拟执行 action，计算新 GRQI，不修改 env
            temp_uav    = list(cur_uav)
            temp_dw_eff = cur_dw_eff.copy()

            if action < P:
                temp_uav.append(action)
            else:
                poi_i = action - P
                row_mean = temp_dw_eff[poi_i].mean()
                temp_dw_eff[poi_i] = (LAMBDA_ENH * row_mean
                                      + (1 - LAMBDA_ENH) * temp_dw_eff[poi_i])

            new_grqi = _compute_grqi(
                env.ftdm,
                _torch.from_numpy(temp_dw_eff.astype('float32')),
                env._D_U,
                temp_uav,
                k_finetune=1,
            )
            delta = new_grqi - cur_grqi
            if delta > best_delta:
                best_delta  = delta
                best_action = action

        state, reward, done, info = env.step(best_action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}
