# rl_extension/ppo_agent.py
# PPO（Proximal Policy Optimization）智能体
#
# 为什么 PPO 比 DQN 更适合本问题：
#   DQN：Q(s,a) 逐个估计每个动作的价值，200 维输出误差积累大
#   PPO：直接优化策略 π(a|s)，通过 Actor-Critic 做信用分配，
#        GAE 估计优势函数，Clip 防止策略更新过大 → 更稳定的收敛
#
# 架构：共享编码器 + Actor 头（动作分布）+ Critic 头（状态价值）
# 关键技术：动作掩码、GAE（广义优势估计）、PPO-Clip 目标函数

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic 网络
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    共享编码器的 Actor-Critic 网络。

    Actor  输出每个动作的 logits，配合动作掩码确保无效动作概率为 0。
    Critic 输出当前状态的价值估计 V(s)，用于 GAE 优势计算。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        # 共享编码器（两层 LayerNorm + ReLU）
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)  # 策略头
        self.critic = nn.Linear(hidden, 1)            # 价值头

        self._init_weights()

    def _init_weights(self):
        """正交初始化，PPO 标准做法，有助于训练初期稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor 小 gain：初始策略接近均匀分布（充分探索）
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Critic 标准 gain
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim)
            mask:  (B, action_dim) bool，True=合法动作
        Returns:
            logits: (B, action_dim)，已屏蔽无效动作
            value:  (B,)
        """
        h      = self.encoder(state)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits, value

    @torch.no_grad()
    def get_action(
        self,
        state: torch.Tensor,
        mask:  torch.Tensor,
    ) -> Tuple[int, float, float]:
        """
        单步采样动作（用于环境交互）。
        Returns: (action, log_prob, value)
        """
        logits, value = self.forward(state.unsqueeze(0), mask.unsqueeze(0))
        logits = logits.squeeze(0)
        value  = value.squeeze(0)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        states:  torch.Tensor,
        actions: torch.Tensor,
        masks:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量重新评估动作（PPO 更新时使用）。
        Returns: (log_probs, values, entropy)
        """
        logits, values = self.forward(states, masks)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ─────────────────────────────────────────────────────────────────────────────
# 轨迹缓冲区（on-policy，每次更新后清空）
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """存储 n 个 episode 的完整轨迹，用于 GAE + PPO 更新。"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []
        self.masks     = []

    def add(self, state, action, log_prob, reward, value, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))
        self.masks.append(mask)

    def compute_gae(
        self,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        广义优势估计（GAE）。

        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        A_t = δ_t + γλ·A_{t+1}   （episode 终止时 A_{T+1}=0）

        优势归一化后方差更小，训练更稳定。
        """
        T       = len(self.rewards)
        values  = torch.tensor(self.values,  dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones   = torch.tensor(self.dones,   dtype=torch.float32)

        advantages = torch.zeros(T)
        last_gae   = 0.0

        for t in reversed(range(T)):
            # episode 终止时，下一步价值为 0
            next_val    = 0.0 if dones[t] else float(values[t + 1]) if t + 1 < T else 0.0
            delta       = rewards[t] + gamma * next_val - values[t]
            last_gae    = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # 优势归一化
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def to_tensors(self):
        return (
            torch.stack(self.states),                              # (T, state_dim)
            torch.tensor(self.actions,   dtype=torch.long),       # (T,)
            torch.tensor(self.log_probs, dtype=torch.float32),    # (T,)
            torch.stack(self.masks),                               # (T, action_dim)
        )

    def __len__(self):
        return len(self.rewards)


# ─────────────────────────────────────────────────────────────────────────────
# PPO 智能体
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO-Clip 智能体。

    超参数说明：
      clip_eps   = 0.2   — 策略比率截断范围，防止单步更新过大
      n_epochs   = 4     — 每批轨迹数据的梯度更新轮数
      vf_coef    = 0.5   — 价值函数损失权重
      ent_coef   = 0.05  — 熵奖励权重（越大越鼓励探索，200 动作空间需要较高值）
      gamma      = 0.99  — 折扣因子
      gae_lambda = 0.95  — GAE λ（经典值，平衡偏差-方差）
    """

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        hidden:      int   = 256,
        lr:          float = 3e-4,
        clip_eps:    float = 0.2,
        n_epochs:    int   = 4,
        gamma:       float = 0.99,
        gae_lambda:  float = 0.95,
        vf_coef:     float = 0.5,
        ent_coef:    float = 0.05,
        max_grad:    float = 0.5,
    ):
        self.policy     = ActorCritic(state_dim, action_dim, hidden)
        self.optimizer  = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps   = clip_eps
        self.n_epochs   = n_epochs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.max_grad   = max_grad
        self.buffer     = RolloutBuffer()

    def get_action(self, state: torch.Tensor, mask: torch.Tensor):
        return self.policy.get_action(state, mask)

    def update(self) -> Dict[str, float]:
        """
        PPO 更新：用当前缓冲区数据做 n_epochs 轮梯度更新，之后清空缓冲区。
        """
        returns, advantages = self.buffer.compute_gae(self.gamma, self.gae_lambda)
        states, actions, old_log_probs, masks = self.buffer.to_tensors()

        logs = {'actor': [], 'value': [], 'entropy': []}

        for _ in range(self.n_epochs):
            log_probs, values, entropy = self.policy.evaluate_actions(
                states, actions, masks
            )

            # 策略比率 r_t = π_new(a|s) / π_old(a|s)
            ratio = (log_probs - old_log_probs).exp()

            # PPO-Clip 目标函数
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值函数 MSE 损失
            value_loss = F.mse_loss(values, returns)

            # 熵奖励（负号因为我们要最大化熵）
            entropy_loss = -entropy.mean()

            loss = (actor_loss
                    + self.vf_coef  * value_loss
                    + self.ent_coef * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
            self.optimizer.step()

            logs['actor'].append(actor_loss.item())
            logs['value'].append(value_loss.item())
            logs['entropy'].append(entropy.mean().item())

        self.buffer.clear()
        return {k: float(np.mean(v)) for k, v in logs.items()}

    def count_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())
