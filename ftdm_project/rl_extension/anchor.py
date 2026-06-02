# rl_extension/anchor.py
# 锚点价值评估模块（Anchor Point Conflict Assessment）
#
# 对应论文开题创新点一：
#   "通过量化分析多维感知数据的方差与冲突度，精准识别出对全局校准
#    价值导向最高的核心区域（锚点），并以此引导无人机进行优先定向采集。"
#
# 锚点分越高 → 该 PoI 人类参与者间分歧越大 → UAV 校验收益越大 → 应优先派遣 UAV

import numpy as np


def compute_anchor_scores(D_W: np.ndarray) -> np.ndarray:
    """
    计算每个 PoI 的综合锚点价值分（Anchor Score），归一化到 [0, 1]。

    三个分量共同量化"方差与冲突度"：
      1. variance_score  = std(D_W[i, :])
                           参与者内部方差：衡量同一 PoI 多个参与者读数的离散程度

      2. conflict_score  = max(D_W[i, :]) - min(D_W[i, :])
                           读数极差（冲突度）：直接刻画参与者之间的最大分歧

      3. deviation_score = |mean(D_W[i, :]) - global_mean|
                           偏离全局均值：识别读数整体偏高/偏低的异常 PoI，
                           这类 PoI 的人类数据系统性偏移最大，UAV 校验价值最高

    综合分 = sum(三分量) 后经 min-max 归一化。
    当所有 PoI 分值相同（无差异）时均返回 0.5（中性先验）。

    Args:
        D_W: (P, I) float32 — P 个 PoI，每 PoI I 名人类参与者的感知值

    Returns:
        anchor_scores: (P,) float32 — 各 PoI 的归一化锚点价值分 ∈ [0, 1]
    """
    D_W = np.asarray(D_W, dtype=np.float32)

    # ── 分量 1：参与者内部方差 ──────────────────────────────────────
    variance_scores = D_W.std(axis=1)               # (P,)

    # ── 分量 2：读数冲突极差 ────────────────────────────────────────
    conflict_scores = D_W.max(axis=1) - D_W.min(axis=1)   # (P,)

    # ── 分量 3：偏离全局均值 ────────────────────────────────────────
    global_mean     = float(D_W.mean())
    deviation_scores = np.abs(D_W.mean(axis=1) - global_mean)  # (P,)

    # ── 综合分（等权求和）──────────────────────────────────────────
    composite = variance_scores + conflict_scores + deviation_scores  # (P,)

    # ── Min-max 归一化到 [0, 1] ─────────────────────────────────────
    lo, hi = composite.min(), composite.max()
    if hi - lo > 1e-8:
        composite = (composite - lo) / (hi - lo)
    else:
        composite = np.full_like(composite, 0.5)   # 所有 PoI 无差异时给中性分

    return composite.astype(np.float32)


def get_top_k_anchors(D_W: np.ndarray, k: int) -> np.ndarray:
    """
    返回锚点分最高的 k 个 PoI 下标（从高到低排序）。
    可用于初始化 UAV 优先访问列表或对比实验。

    Args:
        D_W: (P, I)
        k:   返回的锚点数量

    Returns:
        indices: (k,) int — 按锚点分降序排列的 PoI 下标
    """
    scores = compute_anchor_scores(D_W)
    return np.argsort(scores)[::-1][:k]
