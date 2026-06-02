# data_utils.py
# 数据加载与元学习任务构造
# 严格按照论文 Section VI-A 数据处理方式实现
#
# 论文数据集：UCI 意大利空气质量数据集，9357 条小时均值记录
# 来源注脚1：https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci
#
# 数据分组方式（论文 Section VI-A-2）：
#   将连续数据切分为每组 5 个：4 名人类参与者 + 1 台 UAV
#   D_W: (K, I) — K 组，每组 I=4 名人类参与者的感知值
#   D_U: (K,)   — 每组对应的 UAV 真值

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def load_air_quality(csv_path: str) -> pd.DataFrame:
    """
    加载 UCI 意大利空气质量数据集。
    支持原始分号分隔格式（欧式小数逗号）和标准 CSV 格式。
    异常值 -200 替换为 NaN 后删除。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到数据文件: {csv_path}\n"
            "请先运行 data/download_data.py 下载数据集，或手动放置文件"
        )

    # 尝试多种格式解析
    df = None
    for sep, decimal in [(';', ','), (',', '.'), ('\t', '.')]:
        try:
            tmp = pd.read_csv(csv_path, sep=sep, decimal=decimal)
            # 有效行必须有数值列
            if tmp.shape[1] >= 10:
                df = tmp
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"无法解析数据文件: {csv_path}")

    # 去除列名首尾空格（部分版本有此问题）
    df.columns = [c.strip() for c in df.columns]

    # 用 NaN 替换所有异常值标记 -200
    df = df.replace(Config.OUTLIER_VAL, np.nan)

    # 确认所需列存在
    required = Config.TRAIN_SENSORS + [Config.TEST_SENSOR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"数据集缺少以下列: {missing}\n"
            f"现有列: {list(df.columns)}"
        )

    # 删除任何目标列中含 NaN 的行
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 感知数据矩阵构造（对应论文 Definition 1–3）
# ─────────────────────────────────────────────────────────────────────────────

def build_sensing_matrix(
    series: np.ndarray,
    n_human: int = Config.N_HUMAN,
    group_size: int = Config.GROUP_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将一维时序感知数据切分为感知矩阵。

    论文描述（Section VI-A-2）：
      "we grouped continuous data into sets of five,
       consisting of data from four human participants and one UAV."

    Args:
        series:     一维感知数据数组
        n_human:    每组人类参与者数量（默认 4）
        group_size: 每组总数（默认 5 = 4人 + 1 UAV）

    Returns:
        D_W: (K, n_human) float32 — 人类参与者感知矩阵（论文 Definition 1）
        D_U: (K,)         float32 — UAV 真值向量（Ground Truth）
    """
    n_groups = len(series) // group_size
    if n_groups == 0:
        raise ValueError(f"数据量 {len(series)} 不足以构成一组（需要至少 {group_size}）")

    data = series[:n_groups * group_size].reshape(n_groups, group_size).astype(np.float32)
    D_W = data[:, :n_human]   # 前 n_human 列：人类参与者（有噪声）
    D_U = data[:, n_human]    # 最后一列：UAV（高质量真值）
    return D_W, D_U


def z_score_normalize(
    D_W: np.ndarray,
    D_U: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Z-score 归一化，基于整体均值和方差（含 D_W 和 D_U）。
    返回归一化后的数据和统计量（用于反归一化评估）。
    """
    combined = np.concatenate([D_W.flatten(), D_U])
    mu    = float(combined.mean())
    sigma = float(combined.std()) + 1e-8

    D_W_norm = (D_W - mu) / sigma
    D_U_norm = (D_U - mu) / sigma
    return D_W_norm.astype(np.float32), D_U_norm.astype(np.float32), mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3. 元学习任务构造
# ─────────────────────────────────────────────────────────────────────────────

def make_task(
    D_W: np.ndarray,
    D_U: np.ndarray,
    n_support: int = Config.N_SUPPORT,
    n_query: int   = Config.N_QUERY,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    从感知矩阵中随机切分支撑集（Support Set）和查询集（Query Set）。

    对应 Algorithm 1 第 4 行和第 9 行：
      D_support = {D_W, D_U}（支撑集，用于内循环适应）
      D_query   = {D_W, D_U}（查询集，用于外循环评估）

    Args:
        D_W:       (K, I) 人类参与者感知矩阵
        D_U:       (K,)   UAV 真值向量
        n_support: 支撑集行数 K_s
        n_query:   查询集行数 K_q

    Returns:
        support: {'x': Tensor(K_s, I), 'y': Tensor(K_s,)}
        query:   {'x': Tensor(K_q, I), 'y': Tensor(K_q,)}
    """
    K = len(D_U)
    need = n_support + n_query
    if K < need:
        raise ValueError(
            f"数据量 K={K} 不足，需要至少 {need}（{n_support} 支撑 + {n_query} 查询）"
        )

    idx = np.random.permutation(K)
    sup_idx = idx[:n_support]
    qry_idx = idx[n_support:n_support + n_query]

    support = {
        'x': torch.from_numpy(D_W[sup_idx]),
        'y': torch.from_numpy(D_U[sup_idx]),
    }
    query = {
        'x': torch.from_numpy(D_W[qry_idx]),
        'y': torch.from_numpy(D_U[qry_idx]),
    }
    return support, query


class TaskPool:
    """
    多传感器类型任务池。

    每次调用 sample(M) 返回 M 个随机任务（support + query 对），
    对应论文中 UAV 采集多种数据类型的场景（CO, NO2, 温度等）。

    论文对应：Algorithm 1 中的"M types of tasks T_m"
    """

    def __init__(
        self,
        data_list: List[Tuple[np.ndarray, np.ndarray]],
        n_support: int = Config.N_SUPPORT,
        n_query:   int = Config.N_QUERY,
    ):
        """
        Args:
            data_list: [(D_W_type1, D_U_type1), ...] 各传感器类型的感知矩阵
            n_support: 支撑集大小
            n_query:   查询集大小
        """
        self.data_list = data_list
        self.n_support = n_support
        self.n_query   = n_query

    def sample(self, M: int) -> List[Tuple]:
        """
        随机采样 M 个任务，每个任务来自随机选择的传感器类型。

        对应 Algorithm 1 第 3 行：for i = 1, ..., M do
        """
        tasks = []
        for _ in range(M):
            type_idx = np.random.randint(len(self.data_list))
            D_W, D_U = self.data_list[type_idx]
            support, query = make_task(D_W, D_U, self.n_support, self.n_query)
            tasks.append((support, query))
        return tasks


# ─────────────────────────────────────────────────────────────────────────────
# 4. 小样本微调数据准备（论文 Fine-tuning Process）
# ─────────────────────────────────────────────────────────────────────────────

def prepare_fewshot_data(
    D_W:        np.ndarray,
    D_U:        np.ndarray,
    n_learning: int,
    n_test:     int,
) -> Dict[str, torch.Tensor]:
    """
    为小样本实验准备数据，对应论文 Section VI-C 实验设计。

    场景：
      - n_learning 个 PoI 有 UAV 标定数据（support set，极少量）
      - n_test 个 PoI 仅有人类参与者数据（需预测真值）

    论文实验（Figure 4）：
      n_learning ∈ {6, 7, 8, 9, 10}
      n_test 从 10 × n_learning 开始，步长 10

    Args:
        D_W:        (K, I) 人类参与者感知矩阵
        D_U:        (K,)   UAV 真值
        n_learning: UAV 标定 PoI 数（微调支撑集大小）
        n_test:     待预测 PoI 数（测试集大小）

    Returns:
        dict 包含:
          support_x, support_y — 微调用的极少量 UAV 标定数据
          test_x, test_y       — 测试用的人类感知数据及真值
    """
    K = len(D_U)
    need = n_learning + n_test
    if K < need:
        raise ValueError(
            f"NO_x 数据量 K={K} 不足: 需要 {n_learning} 学习 + {n_test} 测试 = {need}"
        )

    idx      = np.random.permutation(K)
    sup_idx  = idx[:n_learning]
    test_idx = idx[n_learning:n_learning + n_test]

    return {
        'support_x': torch.from_numpy(D_W[sup_idx]),
        'support_y': torch.from_numpy(D_U[sup_idx]),
        'test_x':    torch.from_numpy(D_W[test_idx]),
        'test_y':    torch.from_numpy(D_U[test_idx]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 数据集构建的统一入口
# ─────────────────────────────────────────────────────────────────────────────

def build_all_data(df: pd.DataFrame):
    """
    从数据集 DataFrame 构建所有传感器类型的感知矩阵。

    Returns:
        train_data_list: [(D_W, D_U), ...] 训练传感器类型列表
        test_D_W, test_D_U: NO_x 测试数据（归一化后）
        test_stats: (mu, sigma) 用于反归一化
    """
    # ── 训练任务数据 ─────────────────────────────────────────────────────
    train_data_list = []
    for col in Config.TRAIN_SENSORS:
        series = df[col].values
        D_W, D_U = build_sensing_matrix(series)
        D_W_n, D_U_n, _, _ = z_score_normalize(D_W, D_U)
        train_data_list.append((D_W_n, D_U_n))

    # ── 测试任务数据（NO_x，小样本泛化目标）──────────────────────────────
    nox_series = df[Config.TEST_SENSOR].values
    D_W_test, D_U_test = build_sensing_matrix(nox_series)
    D_W_test_n, D_U_test_n, mu, sigma = z_score_normalize(D_W_test, D_U_test)

    return train_data_list, D_W_test_n, D_U_test_n, (mu, sigma)
