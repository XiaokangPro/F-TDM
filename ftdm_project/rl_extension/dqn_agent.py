# rl_extension/dqn_agent.py
# DQN 智能体实现
#
# 架构：
#   Q 网络: Input(402) → Linear(512,LN,ReLU) → Linear(256,LN,ReLU) → Linear(200)
#   目标网络: 与 Q 网络同架构，每 20 episode 同步一次
#   经验回放缓冲区: maxlen=5000
#   ε-greedy 探索: 1.0 → 0.05（线性衰减，500 episodes）
#   损失函数: Huber Loss（对离群奖励鲁棒）
#   动作掩码: 无效动作 Q 值置 -1e9

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Q 网络
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    DQN Q 函数网络。

    使用 LayerNorm 而非 BatchNorm，保证单样本推理（ε-greedy）时统计量稳定。
    三层结构能够学习 PoI 特征（均值/方差）和分配状态之间的复杂关系。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden1: int = 512, hidden2: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,  hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1,    hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2,    action_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, state_dim)
        Returns:
            q_values: (batch, action_dim)
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# 经验回放缓冲区
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    固定容量的经验回放缓冲区（FIFO）。
    存储 (state, action, reward, next_state, done) 元组。
    """

    def __init__(self, maxlen: int = 5000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool):
        self.buffer.append((
            state.numpy().astype(np.float32),
            int(action),
            float(reward),
            next_state.numpy().astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN 智能体
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN 智能体，支持动作掩码。

    关键设计：
      1. 双网络（Q + target），稳定训练
      2. ε-greedy + 掩码（在合法动作中探索）
      3. Huber Loss（减少大 TD-error 的梯度爆炸）
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        lr:         float = 1e-3,
        gamma:      float = 0.99,
        eps_start:  float = 1.0,
        eps_end:    float = 0.05,
        eps_decay:  int   = 500,     # 线性衰减到 eps_end 所需 episode 数
        buffer_size: int  = 5000,
        batch_size:  int  = 64,
        device:      str  = 'cpu',
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.device     = device

        # ε-greedy 参数
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self._episode   = 0          # 用于计算当前 ε

        # 网络
        self.q_net     = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(maxlen=buffer_size)

        self._train_steps = 0

    @property
    def epsilon(self) -> float:
        """当前探索率（线性衰减）"""
        frac = min(1.0, self._episode / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None) -> int:
        """
        ε-greedy 选动作，配合动作掩码。

        Args:
            state:      (state_dim,) 当前状态
            valid_mask: (action_dim,) bool，True 表示合法动作
        Returns:
            action: int
        """
        if valid_mask is None:
            valid_mask = torch.ones(self.action_dim, dtype=torch.bool)

        valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
        if not valid_indices:
            raise RuntimeError("没有合法动作，请先检查环境状态")

        if random.random() < self.epsilon:
            return random.choice(valid_indices)

        # 贪心：用 Q 网络选动作
        self.q_net.eval()
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(self.device)  # (1, state_dim)
            q_vals  = self.q_net(state_t).squeeze(0)      # (action_dim,)

            # 屏蔽无效动作
            q_vals[~valid_mask] = -1e9
            return int(q_vals.argmax().item())

    def train(self) -> Optional[float]:
        """
        从回放缓冲区采样 batch，更新 Q 网络。

        Returns:
            loss 值（float），若缓冲区样本不足返回 None
        """
        if len(self.memory) < self.batch_size:
            return None

        self.q_net.train()
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # 当前 Q 值
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标 Q 值（Double DQN 思想：用 q_net 选动作，target_net 算值）
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * next_q * (1 - dones)

        # Huber Loss（平滑 L1）
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._train_steps += 1
        return float(loss.item())

    def sync_target(self):
        """将 Q 网络参数同步到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def increment_episode(self):
        """每个 episode 结束时调用，更新 ε"""
        self._episode += 1

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'episode': self._episode,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['q_net'])
        self._episode = ckpt.get('episode', 0)
