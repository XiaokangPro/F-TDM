# config.py
# 论文超参数集中配置
# 来源：论文 Table II (Truth Discovery) 和 Section VI-A (实验设置)


class Config:
    # ── 基础网络结构（论文 Table II）──────────────────────────────────────
    INPUT_SIZE  = 4       # I: 每组人类参与者数（4人 + 1 UAV = 5人/组）
    HIDDEN_SIZE = 1024    # 隐藏层大小

    # ── 元学习超参数（论文 Table II）──────────────────────────────────────
    ALPHA       = 0.0005  # 内循环学习率（论文原文：Inner loop learning rate）
    BETA        = 0.0005  # 外循环学习率（论文原文：Outer loop learning rate）
    INNER_STEPS = 1       # 内循环更新次数（论文原文：Number of inner loop update = 1）
    N_EPOCHS    = 5000    # 元训练轮数（论文原文：Number of training epochs = 5000）
    M_TASKS     = 10      # 每轮外循环采样的任务数 M

    # ── 任务划分──────────────────────────────────────────────────────────
    N_SUPPORT   = 10      # 每个任务的支撑集大小（内循环适应）
    N_QUERY     = 20      # 每个任务的查询集大小（外循环评估）

    # ── 数据处理（论文 Section VI-A-2）────────────────────────────────────
    GROUP_SIZE  = 5       # 每组 = 4 名人类参与者 + 1 台 UAV
    N_HUMAN     = 4       # 每组人类参与者数
    OUTLIER_VAL = -200    # UCI 数据集中异常值标记

    # ── 微调参数（论文 Fine-tuning Process）────────────────────────────────
    FINETUNE_STEPS = 5    # 新任务微调梯度步数

    # ── 数据集列名（论文 Section VI-A-1）──────────────────────────────────
    # 7 种传感器类型：CO, PT08.S1, C6H6, PT08.S2, NO2, temperature, NO_x
    # 训练时排除 NO_x，保留其作为小样本测试的"新任务"
    TRAIN_SENSORS = [
        'CO(GT)',
        'PT08.S1(CO)',
        'C6H6(GT)',
        'PT08.S2(NMHC)',
        'NO2(GT)',
        'T',
    ]
    TEST_SENSOR = 'NOx(GT)'   # 元训练从未见过，用于评估小样本泛化能力

    # ── 小样本实验设置（论文 Section VI-C，Figure 4）────────────────────────
    # 提供给模型的 UAV 标定 PoI 数量（即 support set 大小）
    LEARNING_POIS = [6, 7, 8, 9, 10]

    # ── 其他 ──────────────────────────────────────────────────────────────
    SEED      = 42
    DEVICE    = 'cpu'     # 改为 'cuda' 可使用 GPU（论文用 4×RTX 2080Ti）
    SAVE_DIR  = 'checkpoints'
    SAVE_PATH = 'checkpoints/ftdm_best.pth'
