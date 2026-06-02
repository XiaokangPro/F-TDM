# rl_extension/mcs_env.py
# MCS 联合分配环境（Gym-style）
#
# 问题设定：
#   对 P=100 个 PoI，资源预算 K_total=20 units
#   智能体每步可选择：
#     动作 [0, P)   → 送 UAV 至 PoI i（cost=2 units），获得精确真值作为 F-TDM 支撑集
#     动作 [P, 2P)  → 增强 PoI i-P 的参与者数据（cost=1 unit），降低人类感知噪声
#
# 状态空间（5*P+2 = 502 维）：
#   [is_uav(P), is_enhanced(P), D_W_mean_norm(P), D_W_std_norm(P),
#    anchor_score(P), budget_frac(1), current_grqi(1)]
#
# 多目标奖励（论文开题创新点二）：
#   r_t = α·ΔGRQI - β·能耗代价 + γ·Δ公平性
#   终止时额外 bonus = GRQI_final * 0.1

import numpy as np
import torch
from typing import List, Tuple, Dict

from rl_extension.grqi import compute_grqi, compute_grqi_baseline
from rl_extension.anchor import compute_anchor_scores

# 环境超参数
P_SIZE     = 100    # PoI 子集大小（轻量可运行）
K_TOTAL    = 20     # 总资源预算单位
COST_UAV   = 2      # UAV 动作代价
COST_PART  = 1      # 参与者增强动作代价
LAMBDA_ENH = 0.5    # 参与者增强的平滑系数（λ）
K_FINETUNE = 1      # RL 训练期 GRQI 计算用微调步数（加速）


def _gini(values: np.ndarray) -> float:
    """
    Gini 系数：衡量服务等级分配的不均等程度。
    0 = 完全均等，1 = 极度不均（一个 PoI 独享所有资源）。
    用于量化论文中"参与者任务分配公平性"。
    """
    n = len(values)
    s = float(values.sum())
    if n == 0 or s < 1e-10:
        return 0.0
    sorted_v = np.sort(values)
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, sorted_v) / (n * s)) - (n + 1) / n)


class MCSEnv:
    """
    MCS 联合分配环境。

    核心设计：
      - 每次 reset 从完整测试集随机采样 P 个 PoI 子集，增加训练多样性
      - 动作掩码确保无效动作不被选中
      - 锚点分作为状态特征，引导 agent 学习优先访问高冲突 PoI
      - 多目标奖励：GRQI 提升 + 能耗惩罚 + 公平性奖励
    """

    def __init__(
        self,
        ftdm,
        D_W_pool:   np.ndarray,
        D_U_pool:   np.ndarray,
        p_size:     int   = P_SIZE,
        k_total:    int   = K_TOTAL,
        k_finetune: int   = K_FINETUNE,
        alpha:      float = 1.0,    # GRQI 提升权重
        beta:       float = 0.05,   # 能耗惩罚权重
        gamma:      float = 0.1,    # 公平性奖励权重
    ):
        """
        Args:
            ftdm:       已元训练的 FTDM 实例
            D_W_pool:   (N, I) 完整测试集人类感知数据
            D_U_pool:   (N,)   完整测试集 UAV 真值
            p_size:     每个 episode 使用的 PoI 数量
            k_total:    资源预算总单位数
            k_finetune: F-TDM 微调步数（训练用 1，评估用 5）
            alpha:      GRQI 提升权重（主项）
            beta:       能耗惩罚权重（每步 UAV/增强代价）
            gamma:      公平性奖励权重（Gini 下降即公平提升）
        """
        self.ftdm     = ftdm
        self.D_W_pool = D_W_pool.astype(np.float32)
        self.D_U_pool = D_U_pool.astype(np.float32)
        self.P        = p_size
        self.K_total  = k_total
        self.k_ft     = k_finetune
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma

        self.N          = len(D_U_pool)
        self.state_dim  = 5 * self.P + 2   # 升级：+P 维锚点分
        self.action_dim = 2 * self.P

        # episode 内变量（reset 时重新采样/初始化）
        self._poi_idx:      np.ndarray = None
        self._D_W:          np.ndarray = None
        self._D_U:          np.ndarray = None
        self._D_W_eff:      np.ndarray = None
        self._anchor_scores: np.ndarray = None

        # 归一化统计量（全 pool 计算，episode 内不变）
        self._dw_mean_global = float(self.D_W_pool.mean())
        self._dw_std_global  = float(self.D_W_pool.std()) + 1e-8

        self.reset()

    # ─────────────────────────────────────────────────────────────────
    # 环境接口
    # ─────────────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        """开始新 episode：采样 P 个 PoI 子集，重置状态，计算初始 GRQI/公平度/锚点分。"""
        self._poi_idx = np.random.choice(self.N, self.P, replace=False)
        self._D_W     = self.D_W_pool[self._poi_idx].copy()
        self._D_U     = self.D_U_pool[self._poi_idx].copy()
        self._D_W_eff = self._D_W.copy()

        self._uav_visited: List[int] = []
        self._enhanced:    List[int] = []
        self._budget = self.K_total

        # 锚点分：episode 开始时一次性计算（原始 D_W，不随增强改变）
        self._anchor_scores = compute_anchor_scores(self._D_W)

        # 初始 GRQI 和公平度（无任何分配）
        self._grqi     = compute_grqi_baseline(torch.from_numpy(self._D_W_eff), self._D_U)
        self._fairness = self._compute_fairness()

        return self._get_state()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        执行一步分配动作，返回 (next_state, reward, done, info)。

        多目标奖励（论文开题创新点二）：
          r = α·ΔGRQI - β·energy_cost + γ·Δfairness
        """
        assert self.is_valid(action), f"无效动作: {action}"

        if action < self.P:
            # ── UAV 分配 ────────────────────────────────────────────────
            poi_i = action
            self._uav_visited.append(poi_i)
            self._budget -= COST_UAV
        else:
            # ── 参与者增强 ──────────────────────────────────────────────
            poi_i = action - self.P
            self._enhanced.append(poi_i)
            self._budget -= COST_PART
            row_mean = self._D_W[poi_i].mean()
            self._D_W_eff[poi_i] = (
                LAMBDA_ENH * row_mean + (1 - LAMBDA_ENH) * self._D_W[poi_i]
            )

        # ── 计算新 GRQI ──────────────────────────────────────────────────
        new_grqi = compute_grqi(
            self.ftdm,
            torch.from_numpy(self._D_W_eff),
            self._D_U,
            self._uav_visited,
            k_finetune=self.k_ft,
        )

        # ── 多目标奖励 ───────────────────────────────────────────────────
        delta_grqi  = new_grqi - self._grqi
        step_energy = (COST_UAV if action < self.P else COST_PART) / self.K_total
        new_fairness = self._compute_fairness()
        delta_fair   = new_fairness - self._fairness

        reward = (self.alpha * delta_grqi
                  - self.beta  * step_energy
                  + self.gamma * delta_fair)

        self._grqi     = new_grqi
        self._fairness = new_fairness

        done = self._is_done()
        if done:
            reward += new_grqi * 0.1   # 终止 bonus：鼓励最终高 GRQI

        info = {
            'grqi':        new_grqi,
            'fairness':    new_fairness,
            'n_uav':       len(self._uav_visited),
            'n_enhanced':  len(self._enhanced),
            'budget_left': self._budget,
        }
        return self._get_state(), reward, done, info

    def get_valid_mask(self) -> torch.Tensor:
        """返回 action_dim 长度的布尔掩码（True = 合法动作）。"""
        mask    = torch.zeros(self.action_dim, dtype=torch.bool)
        uav_set = set(self._uav_visited)
        enh_set = set(self._enhanced)

        for i in range(self.P):
            if i not in uav_set and self._budget >= COST_UAV:
                mask[i] = True
            if i not in uav_set and i not in enh_set and self._budget >= COST_PART:
                mask[self.P + i] = True

        return mask

    def is_valid(self, action: int) -> bool:
        return bool(self.get_valid_mask()[action])

    # ─────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────

    def _compute_fairness(self) -> float:
        """
        1 - Gini系数（基于各 PoI 服务等级分布）。
        服务等级：UAV=2，参与者增强=1，未分配=0。
        值越高 = 分配越均等 = 公平性越好。
        """
        service = np.zeros(self.P, dtype=np.float32)
        for i in self._uav_visited:
            service[i] = 2.0
        uav_set = set(self._uav_visited)
        for i in self._enhanced:
            if i not in uav_set:
                service[i] = 1.0
        return 1.0 - _gini(service)

    def _get_state(self) -> torch.Tensor:
        """
        构造 5*P+2 = 502 维状态向量：
          [is_uav(P), is_enhanced(P), dw_mean_norm(P), dw_std_norm(P),
           anchor_score(P), budget_frac(1), current_grqi(1)]
        """
        P = self.P

        is_uav = np.zeros(P, dtype=np.float32)
        for i in self._uav_visited:
            is_uav[i] = 1.0

        is_enh = np.zeros(P, dtype=np.float32)
        for i in self._enhanced:
            is_enh[i] = 1.0

        dw_mean = ((self._D_W_eff.mean(axis=1) - self._dw_mean_global)
                   / self._dw_std_global).astype(np.float32)
        dw_std  = (self._D_W_eff.std(axis=1)
                   / (self._dw_std_global + 1e-8)).astype(np.float32)

        state = np.concatenate([
            is_uav,
            is_enh,
            dw_mean,
            dw_std,
            self._anchor_scores,                          # 新增：锚点价值分
            [self._budget / self.K_total],
            [float(np.clip(self._grqi, -1.0, 1.0))],
        ])
        return torch.from_numpy(state.astype(np.float32))

    def _is_done(self) -> bool:
        if self._budget < COST_PART:
            return True
        return not self.get_valid_mask().any()

    @property
    def current_grqi(self) -> float:
        return self._grqi

    @property
    def current_fairness(self) -> float:
        return self._fairness
