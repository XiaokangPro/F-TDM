# rl_extension/mcs_env.py
# MCS 联合分配环境（Gym-style）
#
# 问题设定：
#   对 P=100 个 PoI，资源预算 K_total=20 units
#   智能体每步可选择：
#     动作 [0, P)   → 送 UAV 至 PoI i（cost=2 units），获得精确真值作为 F-TDM 支撑集
#     动作 [P, 2P)  → 增强 PoI i-P 的参与者数据（cost=1 unit），降低人类感知噪声
#
# 状态空间（402维）：
#   [is_uav(P), is_enhanced(P), D_W_mean_norm(P), D_W_std_norm(P),
#    budget_frac(1), current_grqi(1)]
#
# 奖励：
#   r_t = GRQI_new - GRQI_old  （增量奖励）
#   终止时额外 bonus = GRQI_final * 0.1

import numpy as np
import torch
from typing import List, Tuple, Dict

from rl_extension.grqi import compute_grqi, compute_grqi_baseline

# 环境超参数
P_SIZE      = 100    # PoI 子集大小（轻量可运行）
K_TOTAL     = 20     # 总资源预算单位
COST_UAV    = 2      # UAV 动作代价
COST_PART   = 1      # 参与者增强动作代价
LAMBDA_ENH  = 0.5    # 参与者增强的平滑系数（λ）
K_FINETUNE  = 1      # RL 训练期 GRQI 计算用微调步数（加速）


class MCSEnv:
    """
    MCS 联合分配环境。

    核心设计：
      - 每次 reset 从完整测试集随机采样 P 个 PoI 子集，增加训练多样性
      - 动作掩码确保无效动作不被选中（已 UAV/预算不足等）
      - GRQI 以增量方式计算（减少重复微调）
    """

    def __init__(
        self,
        ftdm,
        D_W_pool:   np.ndarray,   # (N, I) 完整测试集人类感知数据
        D_U_pool:   np.ndarray,   # (N,)   完整测试集 UAV 真值
        p_size:     int = P_SIZE,
        k_total:    int = K_TOTAL,
        k_finetune: int = K_FINETUNE,
    ):
        """
        Args:
            ftdm:      已元训练的 FTDM 实例
            D_W_pool:  (N, I) 完整测试集人类感知数据（每次 reset 从中采样 P 个）
            D_U_pool:  (N,)   完整测试集 UAV 真值
            p_size:    每个 episode 使用的 PoI 数量
            k_total:   资源预算总单位数
            k_finetune: F-TDM 微调步数
        """
        self.ftdm      = ftdm
        self.D_W_pool  = D_W_pool.astype(np.float32)
        self.D_U_pool  = D_U_pool.astype(np.float32)
        self.P         = p_size
        self.K_total   = k_total
        self.k_ft      = k_finetune

        self.N         = len(D_U_pool)
        self.state_dim  = 4 * self.P + 2
        self.action_dim = 2 * self.P

        # episode 内固定的 PoI 子集（reset 时重新采样）
        self._poi_idx: np.ndarray = None
        self._D_W:     np.ndarray = None   # (P, I) 当前 episode 的人类感知数据
        self._D_U:     np.ndarray = None   # (P,)   当前 episode 的 UAV 真值

        # 归一化统计量（用于状态特征）
        self._dw_mean_global = float(self.D_W_pool.mean())
        self._dw_std_global  = float(self.D_W_pool.std()) + 1e-8

        self.reset()

    # ─────────────────────────────────────────────────────────────────
    # 环境接口
    # ─────────────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        """
        开始新的 episode：
          1. 从 pool 随机采样 P 个 PoI 子集
          2. 重置 UAV/增强状态
          3. 计算初始 GRQI（无任何分配）
        """
        self._poi_idx = np.random.choice(self.N, self.P, replace=False)
        self._D_W     = self.D_W_pool[self._poi_idx].copy()   # (P, I)
        self._D_U     = self.D_U_pool[self._poi_idx].copy()   # (P,)

        # 增强后有效感知数据（初始 = 原始数据）
        self._D_W_eff = self._D_W.copy()

        self._uav_visited: List[int] = []    # 已 UAV 标定的 PoI 下标
        self._enhanced:    List[int] = []    # 已增强参与者的 PoI 下标
        self._budget  = self.K_total

        self._grqi = compute_grqi_baseline(
            torch.from_numpy(self._D_W_eff), self._D_U
        )
        return self._get_state()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        执行一步分配动作。

        Args:
            action: int
              [0, P)   → 送 UAV 至 PoI action
              [P, 2P)  → 增强 PoI action-P 的参与者数据

        Returns:
            (next_state, reward, done, info)
        """
        assert self.is_valid(action), f"无效动作: {action}"

        if action < self.P:
            # ── UAV 分配 ────────────────────────────────────────────
            poi_i = action
            self._uav_visited.append(poi_i)
            self._budget -= COST_UAV
        else:
            # ── 参与者增强 ──────────────────────────────────────────
            poi_i = action - self.P
            self._enhanced.append(poi_i)
            self._budget -= COST_PART
            # 对被增强的 PoI 做部分平均降噪
            # D_W_eff[i] = λ*mean + (1-λ)*D_W[i]，保持 I=4 维度不变
            row_mean = self._D_W[poi_i].mean()
            self._D_W_eff[poi_i] = (
                LAMBDA_ENH * row_mean
                + (1 - LAMBDA_ENH) * self._D_W[poi_i]
            )

        # 计算新 GRQI
        new_grqi = compute_grqi(
            self.ftdm,
            torch.from_numpy(self._D_W_eff),
            self._D_U,
            self._uav_visited,
            k_finetune=self.k_ft,
        )

        # 增量奖励
        reward = new_grqi - self._grqi
        self._grqi = new_grqi

        done = self._is_done()
        if done:
            reward += new_grqi * 0.1   # 终止 bonus

        info = {
            'grqi':          new_grqi,
            'n_uav':         len(self._uav_visited),
            'n_enhanced':    len(self._enhanced),
            'budget_left':   self._budget,
        }
        return self._get_state(), reward, done, info

    def get_valid_mask(self) -> torch.Tensor:
        """
        返回 action_dim 长度的布尔掩码，True 表示该动作合法。

        非法条件：
          UAV 动作：PoI 已被 UAV 标定 OR 预算 < COST_UAV
          增强动作：PoI 已被增强 OR PoI 已被 UAV 标定（UAV 更优，无需增强）
                    OR 预算 < COST_PART
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        uav_set  = set(self._uav_visited)
        enh_set  = set(self._enhanced)

        for i in range(self.P):
            # UAV 动作
            if i not in uav_set and self._budget >= COST_UAV:
                mask[i] = True
            # 参与者增强动作
            if (i not in uav_set
                    and i not in enh_set
                    and self._budget >= COST_PART):
                mask[self.P + i] = True

        return mask

    def is_valid(self, action: int) -> bool:
        return bool(self.get_valid_mask()[action])

    # ─────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────

    def _get_state(self) -> torch.Tensor:
        """
        构造 402 维状态向量：
          [is_uav(P), is_enhanced(P), D_W_mean_norm(P), D_W_std_norm(P),
           budget_frac(1), current_grqi(1)]
        """
        P = self.P

        is_uav = np.zeros(P, dtype=np.float32)
        for i in self._uav_visited:
            is_uav[i] = 1.0

        is_enh = np.zeros(P, dtype=np.float32)
        for i in self._enhanced:
            is_enh[i] = 1.0

        # 每 PoI 归一化均值和标准差（让 agent 知道哪些 PoI 不确定性高）
        dw_mean = (self._D_W_eff.mean(axis=1) - self._dw_mean_global) / self._dw_std_global
        dw_std  = self._D_W_eff.std(axis=1) / (self._dw_std_global + 1e-8)

        state = np.concatenate([
            is_uav,
            is_enh,
            dw_mean.astype(np.float32),
            dw_std.astype(np.float32),
            [self._budget / self.K_total],
            [float(np.clip(self._grqi, -1.0, 1.0))],
        ])
        return torch.from_numpy(state.astype(np.float32))

    def _is_done(self) -> bool:
        """当预算耗尽或所有合法动作均不可用时终止"""
        if self._budget < COST_PART:
            return True
        return not self.get_valid_mask().any()

    @property
    def current_grqi(self) -> float:
        return self._grqi
