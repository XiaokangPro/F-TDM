# baselines.py
# 对照基线方法实现
#
# 论文 Section VI-A-5 描述的四种基线（Table II 图4对比）：
#   1. Random parameters ── 随机参数基线（与 F-TDM 同架构但无元训练）
#   2. ITD              ── 激励型真值发现（Incentive-based Truth Discovery）
#   3. TDOD             ── 含离群点检测的真值发现（TDOD [48]）
#   4. Mean             ── 均值算法（论文最弱基线）

import copy
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 基线 1：Mean（均值算法）
# ─────────────────────────────────────────────────────────────────────────────

def mean_predict(D_W: np.ndarray) -> np.ndarray:
    """
    均值算法：对每个 PoI 的所有人类参与者报告值取算术平均。

    论文描述（Section VI-A-5 第4条）：
      "traditional truth discovery methods that lack knowledge of
       the trustworthiness of human participants in a small-sample
       scenario. Therefore, here all human participants are assumed
       the same level of trust."

    这是最朴素的基线，不考虑参与者可靠性差异。

    Args:
        D_W: (K, I) 人类参与者感知矩阵

    Returns:
        pred: (K,) 各 PoI 的均值预测
    """
    return D_W.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 基线 2：Random Parameters
# ─────────────────────────────────────────────────────────────────────────────

def random_predict(model, D_W: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """
    随机参数基线：与 F-TDM 相同网络架构，但参数随机初始化，无元训练。

    论文描述（Section VI-A-5 第1条）：
      "this baseline represents a method without prior learning,
       employing random parameters with the same model and
       learning rate as F-TDM."

    Args:
        model: TruthDiscoveryNet 实例（其参数会被随机重置）
        D_W:   (K, I) 人类感知矩阵 Tensor
        device: 运算设备

    Returns:
        pred: (K,) 随机网络输出
    """
    rand_model = copy.deepcopy(model)
    for layer in rand_model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    rand_model.eval()
    with torch.no_grad():
        return rand_model(D_W.to(device)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 基线 3：ITD（Incentive-based Truth Discovery）
# ─────────────────────────────────────────────────────────────────────────────

def itd_predict(
    D_W_test:   np.ndarray,
    D_W_sup:    np.ndarray,
    D_U_sup:    np.ndarray,
    n_iter:     int = 10,
    eps:        float = 1e-8,
) -> np.ndarray:
    """
    激励型真值发现（ITD）近似实现。

    论文描述（Section VI-A-5 第2条）：
      "the method is employed in handling truth predictions for data
       exclusively submitted by human participants. In the weight
       estimation phase, values provided by the UAVs serve as the
       ground truth for updating the weights of human participants.
       During the truth inference phase, the estimated weights of
       the human participants are used to calculate a weighted
       aggregation calculation on the new sensing data."

    实现流程：
      1. 权重估计阶段（Weight Estimation）：
         用支撑集（UAV 标定数据）迭代估计每名参与者的可信度权重
      2. 真值推断阶段（Truth Inference）：
         用权重对测试集人类数据做加权聚合

    Args:
        D_W_test: (K_test, I) 测试集人类感知矩阵
        D_W_sup:  (K_sup, I)  支撑集人类感知矩阵（有 UAV 标定）
        D_U_sup:  (K_sup,)    支撑集 UAV 真值
        n_iter:   迭代次数
        eps:      数值稳定项

    Returns:
        pred: (K_test,) 预测真值
    """
    I = D_W_sup.shape[1]
    weights = np.ones(I) / I   # 初始均等权重

    # 权重估计阶段：在支撑集上迭代
    for _ in range(n_iter):
        # 用当前权重估计支撑集真值
        truth_est = D_W_sup @ weights   # (K_sup,)

        # 根据每名参与者与真值的误差更新权重
        errors = np.abs(D_W_sup - truth_est[:, None])   # (K_sup, I)
        avg_err = errors.mean(axis=0)                    # (I,)

        # 权重反比于误差（更准确的参与者获得更高权重）
        weights = 1.0 / (avg_err + eps)
        weights = weights / weights.sum()

    # 真值推断阶段：将权重应用到测试集
    return D_W_test @ weights


# ─────────────────────────────────────────────────────────────────────────────
# 基线 4：TDOD（Truth Discovery with Outlier Detection）
# ─────────────────────────────────────────────────────────────────────────────

def tdod_predict(
    D_W_test:   np.ndarray,
    D_W_sup:    np.ndarray,
    D_U_sup:    np.ndarray,
    n_iter:     int   = 10,
    z_threshold: float = 2.0,
    eps:        float  = 1e-8,
) -> np.ndarray:
    """
    含离群点检测的真值发现（TDOD [48]）近似实现。

    论文描述（Section VI-A-5 第3条）：
      "this scheme incorporates outlier detection alongside weight
       and truth update phases to predict the truth for data solely
       submitted by human participants. During the weight update phase,
       human participant weights are adjusted based on ground truth
       information. Subsequently, in the truth update phase, outlier
       detection is initially conducted. Identified outliers are
       eliminated before the updated weights are employed to determine
       the truth for new sensing data."

    实现流程：
      1. 与 ITD 相同的权重估计（使用支撑集）
      2. 对测试集的每个 PoI：
         a. 检测离群参与者（Z-score 超过阈值）
         b. 过滤离群值
         c. 用剩余参与者的归一化权重做加权聚合

    Args:
        D_W_test:    (K_test, I)
        D_W_sup:     (K_sup, I)
        D_U_sup:     (K_sup,)
        n_iter:      权重估计迭代次数
        z_threshold: 离群点判断阈值（Z-score）
        eps:         数值稳定项

    Returns:
        pred: (K_test,) 预测真值
    """
    I = D_W_sup.shape[1]
    weights = np.ones(I) / I

    # Step 1：权重估计（同 ITD）
    for _ in range(n_iter):
        truth_est = D_W_sup @ weights
        errors    = np.abs(D_W_sup - truth_est[:, None])
        avg_err   = errors.mean(axis=0)
        weights   = 1.0 / (avg_err + eps)
        weights   = weights / weights.sum()

    # Step 2：测试集推断（含离群点过滤）
    K_test = D_W_test.shape[0]
    preds  = np.zeros(K_test)

    for i in range(K_test):
        reports = D_W_test[i]

        # 初步估计（加权均值）
        rough_truth = float(reports @ weights)

        # 离群点检测：Z-score
        deviations  = np.abs(reports - rough_truth)
        std         = deviations.std() + eps
        z_scores    = deviations / std
        valid_mask  = z_scores < z_threshold

        if valid_mask.sum() == 0:
            # 所有值都是离群点，退化为均值
            preds[i] = rough_truth
        else:
            # 仅用有效参与者的权重（归一化后）做加权聚合
            valid_w = weights[valid_mask]
            valid_w = valid_w / valid_w.sum()
            preds[i] = float(reports[valid_mask] @ valid_w)

    return preds
