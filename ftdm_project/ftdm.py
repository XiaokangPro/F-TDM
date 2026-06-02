# ftdm.py
# F-TDM：Few-shot Samples Based Truth Discovery Method
#
# 严格按照论文 Section IV 和 Algorithm 1 实现
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  核心思想（论文 Figure 2）                                          │
# │                                                                     │
# │  Training Process（元训练）:                                        │
# │    外循环：在 M 种传感器任务上学会"好的初始化参数 θ"               │
# │    内循环：在每个任务的 Support Set 上做 1 步梯度更新 → θ'          │
# │    外循环目标：θ' 在 Query Set 上表现好 → 反传更新原始 θ           │
# │                                                                     │
# │  Fine-tuning Process（微调）:                                       │
# │    对新任务（如 NO_x），仅用极少 UAV 样本做 k 步内循环              │
# │    → 得到适应后参数，用于校准人类感知数据                            │
# └─────────────────────────────────────────────────────────────────────┘

import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

from config import Config
from model import TruthDiscoveryNet


class FTDM:
    """
    F-TDM: Few-shot Truth Discovery Method

    对应论文 Algorithm 1 和 Figure 2 的完整实现。
    本质是 MAML（Model-Agnostic Meta-Learning）在感知数据真值发现中的应用。
    """

    def __init__(
        self,
        model:       TruthDiscoveryNet,
        alpha:       float = Config.ALPHA,
        beta:        float = Config.BETA,
        inner_steps: int   = Config.INNER_STEPS,
        device:      str   = Config.DEVICE,
    ):
        """
        Args:
            model:       TruthDiscoveryNet 实例（原始参数 θ 存储于此）
            alpha:       内循环学习率（论文 Table II: 0.0005）
            beta:        外循环学习率（论文 Table II: 0.0005）
            inner_steps: 内循环梯度步数（论文 Table II: 1）
            device:      运算设备
        """
        self.model       = model
        self.alpha       = alpha
        self.beta        = beta
        self.inner_steps = inner_steps
        self.device      = device

        self.model.to(self.device)

        # 外循环优化器，用于更新原始参数 θ
        # 对应 Algorithm 1 第 13 行：θ ← θ - β∇_θ L_sum
        self.outer_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.beta
        )

    # ─────────────────────────────────────────────────────────────────────
    # 内循环：在 Support Set 上快速适应
    # ─────────────────────────────────────────────────────────────────────

    def _support_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        支撑集损失函数。
        对应论文公式 (1)：L_support_i = (1/K) ||D_P - D_U||²

        Args:
            pred:   (K,) 模型预测值 D_P
            target: (K,) UAV 真值 D_U
        """
        K = target.shape[0]
        return (1.0 / K) * torch.sum((pred - target) ** 2)

    def _inner_loop(
        self,
        support_x:    torch.Tensor,
        support_y:    torch.Tensor,
        params:       Optional[OrderedDict] = None,
        create_graph: bool = True,
    ) -> Tuple[OrderedDict, torch.Tensor]:
        """
        内循环：在支撑集上做 inner_steps 步梯度更新，得到适应参数 θ'_i。

        对应 Algorithm 1 第 5–8 行：
          5: D_P = f_φ(D_W)
          6: 计算 L_support_i = (1/K)||D_P - D_U||²
          7: 计算 ∇_θ L_support_i
          8: θ'_i = θ - α · ∇_θ L_support_i(f_θ)

        Args:
            support_x:    (K_s, I) 人类感知数据 D_W（支撑集）
            support_y:    (K_s,)   UAV 真值 D_U（支撑集）
            params:       初始参数字典（None 则从模型当前参数克隆）
            create_graph: True = 保留计算图（元训练时必须为 True，
                          使外循环能对θ进行二阶求导）
                          False = 不保留（微调时不需要外循环）

        Returns:
            adapted_params: 更新后的临时参数 θ'_i
            final_loss:     最终一步的支撑集损失（用于监控）
        """
        if params is None:
            params = self.model.get_params()

        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        loss = torch.tensor(0.0)

        for _ in range(self.inner_steps):
            # Step 5：D_P = f_φ(D_W)  使用当前临时参数前向传播
            pred = self.model.functional_forward(support_x, params)

            # Step 6：L_support_i = (1/K)||D_P - D_U||²  论文公式(1)
            loss = self._support_loss(pred, support_y)

            # Step 7：∇_θ L_support_i
            # create_graph=True 时，梯度本身也在计算图中，
            # 支持外循环对 θ 的二阶导数（MAML 核心机制）
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=create_graph,
                allow_unused=True,
            )

            # Step 8：θ'_i = θ - α · ∇_θ L_support_i
            params = OrderedDict(
                (
                    name,
                    param - self.alpha * (
                        grad if grad is not None
                        else torch.zeros_like(param)
                    ),
                )
                for (name, param), grad in zip(params.items(), grads)
            )

        return params, loss

    # ─────────────────────────────────────────────────────────────────────
    # 外循环：元训练一步
    # ─────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        tasks: List[Tuple[Dict, Dict]],
    ) -> float:
        """
        一次外循环步骤，处理 M 个任务。

        对应 Algorithm 1 第 3–13 行（while 循环的一次迭代体）：
          for i = 1, ..., M:
            内循环 → θ'_i
            查询集损失 L_i
          L_sum = Σ L_i
          θ ← θ - β · ∇_θ L_sum

        Args:
            tasks: M 个 (support_dict, query_dict) 元组

        Returns:
            L_sum 的数值（float），用于监控训练过程
        """
        self.outer_optim.zero_grad()

        # L_sum = Σ_{i=1}^{M} L_i(f_{θ'_i})
        L_sum = torch.tensor(0.0, device=self.device)

        for support, query in tasks:
            sup_x = support['x'].to(self.device)
            sup_y = support['y'].to(self.device)
            qry_x = query['x'].to(self.device)
            qry_y = query['y'].to(self.device)

            # 内循环：得到 θ'_i（create_graph=True 保留二阶梯度计算路径）
            # 对应 Algorithm 1 第 4–8 行
            adapted_params, _ = self._inner_loop(
                sup_x, sup_y, create_graph=True
            )

            # Step 10：用 θ'_i 在查询集上计算 L_i
            # 论文原文："Evaluate L_i with weight θ'_i"
            query_pred = self.model.functional_forward(qry_x, adapted_params)
            L_i = nn.MSELoss()(query_pred, qry_y)
            L_sum = L_sum + L_i

        # Step 12–13：∇_θ L_sum 反传，θ ← θ - β · ∇_θ L_sum
        # PyTorch 的 backward() 自动计算二阶梯度（穿透内循环）
        L_sum.backward()
        self.outer_optim.step()

        return float(L_sum.item())

    # ─────────────────────────────────────────────────────────────────────
    # 微调阶段（对应论文 Fine-tuning Process，Figure 2 下半部分）
    # ─────────────────────────────────────────────────────────────────────

    def fine_tune(
        self,
        support_x:  torch.Tensor,
        support_y:  torch.Tensor,
        k_steps:    int = Config.FINETUNE_STEPS,
    ) -> OrderedDict:
        """
        对新任务（如 NO_x）用极少量 UAV 数据做 k 步内循环适应。

        论文描述（Section IV）：
          "F-TDM could calibrate sensing data based solely on human
           participant contributions. By leveraging the meta-learning
           and few-shot learning capabilities of F-TDM, the approach
           enables rapid adaptation to new truth discovery tasks with
           limited data."

        微调无需外循环，不保留计算图（create_graph=False），更快更省内存。

        Args:
            support_x: (n_learning, I) 极少量 UAV 标定数据
            support_y: (n_learning,)   对应 UAV 真值
            k_steps:   微调梯度步数（默认 5）

        Returns:
            adapted_params: 适应后的参数字典，传给 predict() 使用
        """
        params = self.model.get_params()

        for step in range(k_steps):
            params, loss = self._inner_loop(
                support_x,
                support_y,
                params=params,
                create_graph=False,   # 微调只需一阶梯度
            )

        return params

    # ─────────────────────────────────────────────────────────────────────
    # 推理
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x:      torch.Tensor,
        params: Optional[OrderedDict] = None,
    ) -> torch.Tensor:
        """
        预测真值。

        Args:
            x:      (N, I) 人类参与者感知数据
            params: 适应后的参数（None = 使用原始 θ，适用于元训练评估）

        Returns:
            pred: (N,) 预测真值
        """
        x = x.to(self.device)
        if params is None:
            return self.model(x)
        return self.model.functional_forward(x, params)

    # ─────────────────────────────────────────────────────────────────────
    # 模型保存 / 加载
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'alpha':       self.alpha,
                'beta':        self.beta,
                'inner_steps': self.inner_steps,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.alpha       = ckpt.get('alpha',       self.alpha)
        self.beta        = ckpt.get('beta',        self.beta)
        self.inner_steps = ckpt.get('inner_steps', self.inner_steps)
