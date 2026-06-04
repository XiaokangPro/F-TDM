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
#   5. Energy-Min       — 优先选数据值最相近的 PoI（模拟 UAV 就近飞行，最小化能耗）
#                         代表传统 MCS 调度"只优化飞行成本、不考虑数据质量"的范式

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


# ─────────────────────────────────────────────────────────────────────────────
# 基线 5：Energy-Min（最小能耗，就近选点）
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_energy_min(env) -> Dict:
    """
    最小能耗基线：UAV 优先访问数据值最相似（"距离最近"）的 PoI 群组。

    建模依据：
      真实场景中 UAV 能耗正比于飞行距离，空间相邻的 PoI 数据值通常相关
      （地理空间自相关性）。用 D_W 行均值作为"地理位置"的代理：
        → 数据值相近的 PoI 视为"空间邻近"
        → 最小能耗 = 集中访问数据值相近的一组 PoI（本地巡逻路线）

    关键局限：
      这种就近集中策略使支撑集多样性极低，F-TDM 微调后只能精确校准
      与支撑集相似的 PoI，对其他区域泛化能力差 → GRQI 显著下降。
      这正说明"只优化飞行成本、不考虑数据校准质量"的传统调度范式的局限。

    对比意义：
      与随机分配对比：验证盲目最小化能耗会损害数据恢复质量
      与 RL-GRQI 对比：验证引入质量反馈的必要性
    """
    from rl_extension.mcs_env import COST_UAV, COST_PART

    P     = env.P
    state = env.reset()
    done  = False
    total_reward = 0.0
    info  = {'n_uav': 0, 'n_enhanced': 0, 'budget_left': env.K_total}

    # 按 D_W 行均值排序（升序），模拟沿数据梯度方向的"地理序"
    dw_means   = env._D_W.mean(axis=1)        # (P,)
    sorted_idx = np.argsort(dw_means)          # 数据值从小到大的 PoI 下标

    # 最小能耗策略：从排好序的 PoI 中取最"集中"（相邻）的一段作为 UAV 目标
    # 预算可飞的最大 UAV 次数
    max_uav = env.K_total // COST_UAV          # e.g. 20//2 = 10
    # 取排序后最中间的 max_uav 个 PoI（最聚集、路程最短）
    mid   = P // 2
    half  = max_uav // 2
    start = max(0, mid - half)
    uav_targets = set(sorted_idx[start: start + max_uav].tolist())

    while not done:
        mask  = env.get_valid_mask()
        valid = mask.nonzero(as_tuple=True)[0].tolist()
        if not valid:
            break

        # 优先执行预定 UAV 目标
        uav_valid = [i for i in valid if i < P and i in uav_targets]
        if uav_valid:
            # 按"就近"顺序（sorted_idx 中的顺序）执行
            action = min(uav_valid,
                         key=lambda i: np.where(sorted_idx == i)[0][0])
        else:
            # UAV 预算用完或目标已全部访问，剩余预算随机增强参与者
            part_valid = [i for i in valid if i >= P]
            if not part_valid:
                break
            action = random.choice(part_valid)

        state, reward, done, info = env.step(action)
        total_reward += reward

    return {'grqi': env.current_grqi, 'reward': total_reward,
            'n_uav': info['n_uav'], 'n_enhanced': info['n_enhanced']}
