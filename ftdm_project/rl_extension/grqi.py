# rl_extension/grqi.py
# 全局恢复精度指标（Global Recovery Quality Index, GRQI）
#
# 定义：
#   GRQI = R²(F-TDM预测结果, 非UAV-PoI真值)
#        = 1 - Σ(ŷ - y)² / Σ(y - ȳ)²
#
# 取值范围：(-∞, 1]
#   1.0 = 完美预测（F-TDM 准确还原所有非UAV PoI 的真值）
#   0.0 = 预测水平等同于全局均值（无信息增益）
#  <0   = 预测质量差于均值基线
#
# 设计原则：
#   - 仅评估「非 UAV 标定」的 PoI，UAV PoI 质量已知为 1.0（不计入以防信息泄漏）
#   - k_finetune=1（RL 训练时快速估算奖励）
#   - k_finetune=5（评估/对比时精确计算）

import numpy as np
import torch
from typing import List


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    决定系数 R²（Coefficient of Determination）

    R² = 1 - SS_res / SS_tot
    SS_res = Σ(y_true - y_pred)²
    SS_tot = Σ(y_true - mean(y_true))²
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_grqi(
    ftdm,
    all_D_W:     torch.Tensor,
    all_D_U_np:  np.ndarray,
    uav_indices: List[int],
    k_finetune:  int = 1,
) -> float:
    """
    计算全局恢复精度 GRQI。

    Args:
        ftdm:        已元训练的 FTDM 实例
        all_D_W:     (P, I) Tensor，全部 PoI 的人类感知矩阵（可能已增强）
        all_D_U_np:  (P,) numpy，全部 PoI 的 UAV 真值（仅用于 GRQI 计算，不暴露给 agent）
        uav_indices: UAV 已标定的 PoI 下标列表（F-TDM 微调用的支撑集）
        k_finetune:  F-TDM 微调梯度步数
                     RL 训练时用 1（快速），评估时用 5（精确）

    Returns:
        GRQI ∈ (-inf, 1.0]
    """
    P = len(all_D_U_np)
    uav_set = set(uav_indices)
    non_uav = [i for i in range(P) if i not in uav_set]

    if len(non_uav) == 0:
        return 1.0   # 所有 PoI 均已 UAV 标定，质量满分

    test_x = all_D_W[non_uav]          # (n_non_uav, I)
    test_y = all_D_U_np[non_uav]       # (n_non_uav,)

    if len(uav_indices) == 0:
        # 无 UAV 支撑集：退化为均值基线预测
        preds = all_D_W[non_uav].numpy().mean(axis=1)
        return r2_score(test_y, preds)

    # 用 UAV 标定数据微调 F-TDM
    sup_x = all_D_W[uav_indices]                          # (n_uav, I)
    sup_y = torch.from_numpy(all_D_U_np[uav_indices])     # (n_uav,)

    adapted_params = ftdm.fine_tune(sup_x, sup_y, k_steps=k_finetune)

    # 对非 UAV PoI 预测真值
    preds = ftdm.predict(test_x, adapted_params).cpu().numpy()

    return r2_score(test_y, preds)


def compute_grqi_baseline(all_D_W: torch.Tensor, all_D_U_np: np.ndarray) -> float:
    """
    零分配基线 GRQI：没有任何 UAV，仅用人类均值预测。
    用于计算 GRQI 提升幅度的参照点。
    """
    preds = all_D_W.numpy().mean(axis=1)
    return r2_score(all_D_U_np, preds)
