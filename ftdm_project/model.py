# model.py
# TruthDiscoveryNet：F-TDM 的基础网络
#
# 论文描述（Section IV, Table II）：
#   全连接神经网络，隐藏层大小 1024
#   输入：D_W 中一行（I=4 个人类参与者的感知值）
#   输出：该 PoI 的估计真值（标量）
#
# 关键设计：实现 functional_forward 方法
#   MAML 内循环需要使用"临时参数 θ'_i"做前向传播，
#   而 θ'_i 不在 self.parameters() 中，必须显式传参。

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from config import Config


class TruthDiscoveryNet(nn.Module):
    """
    F-TDM 基础网络：两层隐藏层的全连接网络。

    网络结构：
        Input(I) → Linear → ReLU → Linear → ReLU → Linear → Output(1)

    支持两种前向传播模式：
      1. forward(x)               — 标准模式，使用 self 中的参数（推理、外循环）
      2. functional_forward(x, p) — 参数字典模式，使用传入的参数（MAML 内循环）
    """

    def __init__(
        self,
        input_size:  int = Config.INPUT_SIZE,
        hidden_size: int = Config.HIDDEN_SIZE,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，与论文"randomly initialize parameters"一致"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ── 标准前向传播（外循环 / 推理时使用）──────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, I) 人类参与者感知值
        Returns:
            pred: (batch,) 预测真值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

    # ── 参数字典前向传播（MAML 内循环专用）──────────────────────────────────

    def functional_forward(
        self,
        x:      torch.Tensor,
        params: OrderedDict,
    ) -> torch.Tensor:
        """
        使用外部传入的参数字典做前向传播。

        MAML 内循环更新得到的临时参数 θ'_i 并不存储在 self 中，
        必须通过此方法才能让前向传播走 θ'_i，同时保留计算图
        用于外循环的二阶梯度计算。

        Args:
            x:      (batch, I) 人类参与者感知值
            params: OrderedDict，key 为 'fc1.weight'/'fc1.bias'/...

        Returns:
            pred: (batch,) 预测真值
        """
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
        x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x.squeeze(-1)

    # ── 参数辅助方法 ──────────────────────────────────────────────────────

    def get_params(self) -> OrderedDict:
        """
        返回当前参数的克隆字典。

        clone() 会保留 requires_grad=True，且创建新的张量节点，
        使内循环的梯度计算可以独立于原始参数 θ 进行。

        对应 Algorithm 1 第 2 行：F(φ) 复制为 φ'（临时参数的初始值）
        """
        return OrderedDict(
            (name, param.clone())
            for name, param in self.named_parameters()
        )

    def count_params(self) -> int:
        """统计可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
