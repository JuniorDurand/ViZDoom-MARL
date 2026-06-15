"""
Recurrent Actor-Critic for PPO.
CNN encoder → GRUCell → actor head (policy) + critic head (value).

Uses GRUCell instead of GRU because PPO collects data online,
processing one timestep at a time during rollout collection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.base import BaseNetwork


class RecurrentActorCritic(BaseNetwork):

    has_memory = True

    def __init__(self, in_channels: int, n_actions: int, hidden_size: int = 256):
        super().__init__(in_channels, n_actions, hidden_size)

        self.fc = nn.Linear(self._encoder_out_size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()
        # smaller init for actor (encourages exploration early on)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs, hidden=None):
        """
        obs: (batch, C, H, W) — single timestep
        hidden: (batch, hidden_size)
        Returns: logits (batch, n_actions), value (batch,), hidden (batch, hidden_size)
        """
        x = self.encode(obs)
        x = F.relu(self.fc(x), inplace=True)

        if hidden is None:
            hidden = self.init_hidden(obs.size(0), obs.device)

        hidden = self.gru_cell(x, hidden)

        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)

        return logits, value, hidden

    def init_hidden(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)