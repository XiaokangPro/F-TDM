# evaluate.py
# 评估模块：实现论文 Section VI 的实验指标和评估流程
#
# 指标（论文 Section VI-A-4）：
#   RMSE = sqrt(1/N * Σ(y_i - ŷ_i)²)     论文公式
#   MAPE = (1/N)*Σ|y_i-ŷ_i|/|y_i| × 100%  论文公式
#
# 实验流程（论文 Section VI-C，Figure 4）：
#   对 n_learning ∈ {6,7,8,9,10}，
#   测试在 10×n_learning 到 (10×n_learning + 50) 个 PoI 上的 RMSE

import numpy as np
import torch
from typing import Dict, List, Tuple

from config import Config
from data_utils import prepare_fewshot_data
from baselines import mean_predict, random_predict, itd_predict, tdod_predict


# ─────────────────────────────────────────────────────────────────────────────
# 指标函数
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = sqrt(1/N * Σ(y_i - ŷ_i)²)"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """MAPE = (1/N)*Σ|y_i - ŷ_i|/|y_i| * 100%"""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# 单次小样本评估（重复 n_runs 次取平均）
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fewshot(
    ftdm,
    D_W:        np.ndarray,
    D_U:        np.ndarray,
    n_learning: int,
    n_test:     int,
    n_runs:     int = 5,
) -> Dict[str, float]:
    """
    小样本评估：用 n_learning 个 UAV 标定样本微调，在 n_test 个 PoI 上测试。

    对应论文 Section VI-C 实验流程：
      - n_learning 为 support set 大小（UAV 提供的标定 PoI）
      - n_test 为需要校准真值的人类感知 PoI 数量
      - 重复 n_runs 次随机采样取 RMSE 均值

    Args:
        ftdm:       已训练的 FTDM 实例
        D_W:        (K, I) NO_x 人类感知矩阵（测试数据集）
        D_U:        (K,)   NO_x UAV 真值
        n_learning: UAV 标定 PoI 数（微调支撑集大小）
        n_test:     待预测 PoI 数
        n_runs:     重复实验次数（取均值降低随机性影响）

    Returns:
        {方法名: RMSE均值}
    """
    all_rmse: Dict[str, List[float]] = {
        'F-TDM':  [],
        'Mean':   [],
        'Random': [],
        'ITD':    [],
        'TDOD':   [],
    }

    for _ in range(n_runs):
        data = prepare_fewshot_data(D_W, D_U, n_learning, n_test)

        sup_x  = data['support_x']             # Tensor (n_learning, I)
        sup_y  = data['support_y']             # Tensor (n_learning,)
        tst_x  = data['test_x']               # Tensor (n_test, I)
        tst_y  = data['test_y'].numpy()        # np (n_test,) ← 真值

        sup_xnp = sup_x.numpy()               # np (n_learning, I)
        sup_ynp = sup_y.numpy()               # np (n_learning,)
        tst_xnp = tst_x.numpy()              # np (n_test, I)

        # ── F-TDM：微调后预测 ─────────────────────────────────────────
        adapted = ftdm.fine_tune(sup_x, sup_y)
        ftdm_pred = ftdm.predict(tst_x, adapted).cpu().numpy()
        all_rmse['F-TDM'].append(rmse(tst_y, ftdm_pred))

        # ── Mean 基线 ────────────────────────────────────────────────
        mean_pred = mean_predict(tst_xnp)
        all_rmse['Mean'].append(rmse(tst_y, mean_pred))

        # ── Random 基线 ──────────────────────────────────────────────
        rand_pred = random_predict(ftdm.model, tst_x, ftdm.device)
        all_rmse['Random'].append(rmse(tst_y, rand_pred))

        # ── ITD 基线 ─────────────────────────────────────────────────
        itd_pred = itd_predict(tst_xnp, sup_xnp, sup_ynp)
        all_rmse['ITD'].append(rmse(tst_y, itd_pred))

        # ── TDOD 基线 ────────────────────────────────────────────────
        tdod_pred = tdod_predict(tst_xnp, sup_xnp, sup_ynp)
        all_rmse['TDOD'].append(rmse(tst_y, tdod_pred))

    return {k: float(np.mean(v)) for k, v in all_rmse.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 完整论文实验：复现 Figure 4
# ─────────────────────────────────────────────────────────────────────────────

def run_paper_experiment(
    ftdm,
    D_W_test:      np.ndarray,
    D_U_test:      np.ndarray,
    learning_pois: List[int] = None,
    n_runs:        int       = 5,
) -> Dict:
    """
    复现论文 Figure 4 实验。

    对于每个 n_learning ∈ {6,7,8,9,10}：
      测试集大小从 10×n_learning 开始，步长 10，共 6 个测试点
      (Figure 4(a)~(e) 各子图的 x 轴)

    Returns:
        results[n_learning][n_test] = {'F-TDM': rmse, 'Mean': rmse, ...}
    """
    if learning_pois is None:
        learning_pois = Config.LEARNING_POIS

    results = {}
    K = len(D_U_test)

    for n_learning in learning_pois:
        results[n_learning] = {}
        n_test_start = n_learning * 10
        # 从 10x 开始，步长 10，共取 6 个点（对应 Figure 4 x 轴）
        n_test_list  = list(range(n_test_start, n_test_start + 60, 10))

        print(f"\n  ── n_learning = {n_learning} 个 UAV PoI ──────────────────")

        for n_test in n_test_list:
            if n_learning + n_test > K:
                print(f"    [跳过] 数据不足: 需要 {n_learning+n_test}, 仅有 {K}")
                continue

            res = evaluate_fewshot(
                ftdm, D_W_test, D_U_test,
                n_learning, n_test, n_runs
            )
            results[n_learning][n_test] = res

            # 格式化输出（对齐列，便于和论文 Figure 4 对比）
            line = (
                f"    n_test={n_test:4d} | "
                f"F-TDM={res['F-TDM']:6.3f}  "
                f"Mean={res['Mean']:6.3f}  "
                f"ITD={res['ITD']:6.3f}  "
                f"TDOD={res['TDOD']:6.3f}  "
                f"Random={res['Random']:6.3f}"
            )
            print(line)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 论文 Figure 4(f)：固定 100 个测试 PoI，变化 n_learning（5~10）
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_test_experiment(
    ftdm,
    D_W_test:   np.ndarray,
    D_U_test:   np.ndarray,
    n_test:          int        = 100,
    n_learning_list: List[int] = None,
    n_runs:          int       = 5,
) -> Dict:
    """
    固定测试 PoI 数量（论文设 100），变化 n_learning 从 5 到 10。
    对应论文 Figure 4(f)。

    Returns:
        results[n_learning] = {方法: RMSE}
    """
    if n_learning_list is None:
        n_learning_list = list(range(5, 11))   # 5, 6, 7, 8, 9, 10

    K = len(D_U_test)
    results = {}

    print(f"\n  ── 固定 n_test={n_test}，变化 n_learning（Figure 4f）──────────")

    for n_learning in n_learning_list:
        if n_learning + n_test > K:
            continue

        res = evaluate_fewshot(
            ftdm, D_W_test, D_U_test,
            n_learning, n_test, n_runs
        )
        results[n_learning] = res
        line = (
            f"    n_learning={n_learning:2d} | "
            f"F-TDM={res['F-TDM']:6.3f}  "
            f"Mean={res['Mean']:6.3f}  "
            f"ITD={res['ITD']:6.3f}  "
            f"TDOD={res['TDOD']:6.3f}  "
            f"Random={res['Random']:6.3f}"
        )
        print(line)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict, title: str = "实验结果汇总"):
    """打印实验结果汇总，并验证 F-TDM 是否优于所有基线"""
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)

    methods    = ['F-TDM', 'Mean', 'ITD', 'TDOD', 'Random']
    wins       = {m: 0 for m in methods if m != 'F-TDM'}
    total_cmp  = 0

    for k1, v1 in results.items():
        if isinstance(v1, dict) and isinstance(list(v1.values())[0], dict):
            # 两层结构：results[n_learning][n_test]
            for k2, scores in v1.items():
                total_cmp += 1
                ftdm_s = scores.get('F-TDM', float('inf'))
                for m in wins:
                    if ftdm_s < scores.get(m, float('inf')):
                        wins[m] += 1
        else:
            # 单层结构：results[n_learning]
            scores = v1
            total_cmp += 1
            ftdm_s = scores.get('F-TDM', float('inf'))
            for m in wins:
                if ftdm_s < scores.get(m, float('inf')):
                    wins[m] += 1

    print("F-TDM 优于各基线的比例：")
    for m, w in wins.items():
        pct = w / total_cmp * 100 if total_cmp else 0
        print(f"  vs {m:<8s}: {w}/{total_cmp} = {pct:.1f}%")

    print()
