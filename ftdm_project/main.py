#!/usr/bin/env python3
# main.py
# F-TDM 程序总入口
#
# 使用方法：
#   完整训练（5000 轮，论文参数）：
#     python main.py --data_path data/AirQualityUCI.csv
#
#   快速验证（200 轮，适合先确认代码正确）：
#     python main.py --data_path data/AirQualityUCI.csv --quick
#
#   跳过训练，加载已保存模型：
#     python main.py --data_path data/AirQualityUCI.csv --load_model
#
# 数据集下载：
#     python data/download_data.py

import os
import sys
import argparse
import random
import numpy as np
import torch

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_utils import load_air_quality, build_all_data, TaskPool
from model import TruthDiscoveryNet
from ftdm import FTDM
from train import train, quick_train
from evaluate import (
    run_paper_experiment,
    run_fixed_test_experiment,
    print_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed()
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    # ── 步骤 1：加载数据集 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1  加载数据集")
    print("=" * 60)

    df = load_air_quality(args.data_path)
    print(f"  数据集路径: {args.data_path}")
    print(f"  有效记录数: {len(df)} 条")
    print(f"  元训练传感器: {Config.TRAIN_SENSORS}")
    print(f"  小样本测试传感器（新任务）: {Config.TEST_SENSOR}")

    # ── 步骤 2：构建数据矩阵和任务池 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2  构建感知矩阵和元学习任务池")
    print("=" * 60)

    train_data_list, D_W_test, D_U_test, (mu, sigma) = build_all_data(df)

    print(f"  训练任务数（传感器类型）: {len(train_data_list)}")
    print(f"  各训练任务 PoI 数: "
          f"{[d[0].shape[0] for d in train_data_list]}")
    print(f"  NO_x 测试 PoI 数: {D_W_test.shape[0]}")
    print(f"  每 PoI 人类参与者数 I: {D_W_test.shape[1]}")

    task_pool = TaskPool(
        data_list = train_data_list,
        n_support = Config.N_SUPPORT,
        n_query   = Config.N_QUERY,
    )

    # ── 步骤 3：元训练或加载已有模型 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3  F-TDM 元训练（对应论文 Algorithm 1）")
    print("=" * 60)

    save_path = Config.SAVE_PATH

    if args.load_model and os.path.exists(save_path):
        print(f"  检测到已保存模型，直接加载: {save_path}")
        model = TruthDiscoveryNet(Config.INPUT_SIZE, Config.HIDDEN_SIZE)
        ftdm  = FTDM(model)
        ftdm.load(save_path)

    elif args.quick:
        print("  快速验证模式（200 轮，用于检验代码流程）")
        ftdm = quick_train(task_pool, n_epochs=200, M=5)

    else:
        print(f"  完整元训练（{args.epochs} 轮，论文参数）")
        ftdm = train(
            task_pool = task_pool,
            n_epochs  = args.epochs,
            M         = args.M,
            save_path = save_path,
        )

    # ── 步骤 4：小样本实验（复现 Figure 4(a)~(e)）────────────────────────
    print("\n" + "=" * 60)
    print("Step 4  小样本实验（复现论文 Figure 4(a)~(e)）")
    print("=" * 60)
    print("  实验设置：")
    print("    - 对每个 n_learning ∈ {6,7,8,9,10}，")
    print("    - 从 10×n_learning 开始，步长 10，测试 6 个 n_test 值")
    print("    - 每组配置重复 5 次取 RMSE 均值")
    print()

    results_fig4 = run_paper_experiment(
        ftdm      = ftdm,
        D_W_test  = D_W_test,
        D_U_test  = D_U_test,
        n_runs    = 5,
    )

    # ── 步骤 5：固定测试集实验（复现 Figure 4(f)）────────────────────────
    print("\n" + "=" * 60)
    print("Step 5  固定 100 个测试 PoI，变化 n_learning（Figure 4f）")
    print("=" * 60)

    results_fig4f = run_fixed_test_experiment(
        ftdm      = ftdm,
        D_W_test  = D_W_test,
        D_U_test  = D_U_test,
        n_test    = 100,
        n_runs    = 5,
    )

    # ── 步骤 6：汇总输出 ──────────────────────────────────────────────────
    print_summary(results_fig4,  title="Figure 4(a)~(e) 结果汇总")
    print_summary(results_fig4f, title="Figure 4(f) 结果汇总")

    print("\n  实验完成！")
    print("  核心结论：F-TDM 在小样本场景下应优于所有基线。")
    print("  若 F-TDM 未明显领先，可增加训练轮数（--epochs 5000）。\n")

    return results_fig4, results_fig4f


# ─────────────────────────────────────────────────────────────────────────────
# CLI 参数
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='F-TDM: Few-shot Truth Discovery Method (论文复现)'
    )

    parser.add_argument(
        '--data_path', type=str,
        default='data/AirQualityUCI.csv',
        help='UCI 意大利空气质量数据集路径（默认 data/AirQualityUCI.csv）'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=Config.N_EPOCHS,
        help=f'元训练轮数（论文: {Config.N_EPOCHS}，默认: {Config.N_EPOCHS}）'
    )
    parser.add_argument(
        '--M', type=int,
        default=Config.M_TASKS,
        help=f'每轮任务数 M（默认: {Config.M_TASKS}）'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速验证模式（200轮），用于检查代码是否可以正常运行'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help=f'加载已保存模型（{Config.SAVE_PATH}），跳过训练'
    )
    parser.add_argument(
        '--device', type=str,
        default=Config.DEVICE,
        choices=['cpu', 'cuda'],
        help='运算设备（默认 cpu，有 GPU 建议改为 cuda）'
    )

    args = parser.parse_args()

    # 允许命令行覆盖 device 配置
    Config.DEVICE = args.device

    main(args)
