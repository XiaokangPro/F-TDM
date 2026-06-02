# train.py
# F-TDM 元训练主循环
# 对应论文 Algorithm 1 的完整 while 循环

import os
import time
import torch
import numpy as np
from typing import Optional

from config import Config
from model import TruthDiscoveryNet
from ftdm import FTDM
from data_utils import TaskPool


def train(
    task_pool: TaskPool,
    n_epochs:  int  = Config.N_EPOCHS,
    M:         int  = Config.M_TASKS,
    save_path: str  = Config.SAVE_PATH,
    log_every: int  = 500,
    verbose:   bool = True,
) -> FTDM:
    """
    F-TDM 元训练主循环。

    对应论文 Algorithm 1：
      while not reach epochs do
        采样 M 个任务 → 内循环 → 查询集损失 → 外循环更新 θ
      end while

    训练后模型具备"快速适应新任务"的能力：
    面对从未见过的 NO_x 传感器数据，只需 6~10 个 UAV 标定样本
    即可通过几步微调达到良好性能。

    Args:
        task_pool: TaskPool 实例（包含所有训练传感器类型的数据）
        n_epochs:  元训练轮数（论文 Table II: 5000）
        M:         每轮采样任务数（论文原文 M）
        save_path: 最优模型保存路径
        log_every: 日志打印间隔（轮数）
        verbose:   是否打印训练日志

    Returns:
        trained FTDM instance
    """
    model = TruthDiscoveryNet(
        input_size  = Config.INPUT_SIZE,
        hidden_size = Config.HIDDEN_SIZE,
    )
    ftdm = FTDM(model)

    if verbose:
        total_params = model.count_params()
        print("=" * 60)
        print("F-TDM 元训练启动")
        print(f"  网络参数量:       {total_params:,}")
        print(f"  内循环学习率 α:   {ftdm.alpha}")
        print(f"  外循环学习率 β:   {ftdm.beta}")
        print(f"  内循环步数:       {ftdm.inner_steps}")
        print(f"  元训练轮数:       {n_epochs}")
        print(f"  每轮任务数 M:     {M}")
        print(f"  运算设备:         {ftdm.device}")
        print("=" * 60)

    best_loss  = float('inf')
    loss_log   = []
    t_start    = time.time()

    for epoch in range(1, n_epochs + 1):

        # ── 采样 M 个任务（对应 Algorithm 1 第 3 行：for i=1,...,M）─────
        tasks = task_pool.sample(M)

        # ── 外循环一步（Algorithm 1 第 5-13 行）──────────────────────────
        loss = ftdm.train_step(tasks)
        loss_log.append(loss)

        # ── 保存最优模型 ─────────────────────────────────────────────────
        if loss < best_loss:
            best_loss = loss
            ftdm.save(save_path)

        # ── 日志输出 ─────────────────────────────────────────────────────
        if verbose and epoch % log_every == 0:
            elapsed   = time.time() - t_start
            avg_loss  = float(np.mean(loss_log[-log_every:]))
            eta_sec   = elapsed / epoch * (n_epochs - epoch)
            print(
                f"Epoch {epoch:5d}/{n_epochs}  "
                f"L_sum={loss:.6f}  "
                f"avg({log_every})={avg_loss:.6f}  "
                f"best={best_loss:.6f}  "
                f"elapsed={elapsed:.0f}s  ETA={eta_sec:.0f}s"
            )

    if verbose:
        total_time = time.time() - t_start
        print("=" * 60)
        print(f"元训练完成！总耗时: {total_time:.1f}s")
        print(f"最优 L_sum: {best_loss:.6f}")
        print(f"模型已保存至: {save_path}")
        print("=" * 60)

    # 加载最优参数返回
    ftdm.load(save_path)
    return ftdm


def quick_train(
    task_pool: TaskPool,
    n_epochs:  int = 200,
    M:         int = 5,
    verbose:   bool = True,
) -> FTDM:
    """
    快速测试模式（少轮次），用于验证代码流程正确性。
    与完整训练参数相同，但仅跑 n_epochs 轮。
    """
    return train(
        task_pool = task_pool,
        n_epochs  = n_epochs,
        M         = M,
        save_path = 'checkpoints/ftdm_quick.pth',
        log_every = max(1, n_epochs // 4),
        verbose   = verbose,
    )
