"""
DQN: CNN encoder → FC → Q-values.
No temporal memory. Uses frame stacking for context.
"""

import torch
import torch.nn as nn

from src.networks.base import BaseNetwork


class DQNetwork(BaseNetwork):

    has_memory = False

    def __init__(self, in_channels: int, n_actions: int, hidden_size: int = 256):
        super().__init__(in_channels, n_actions, hidden_size)

        self.head = nn.Sequential(
            nn.Linear(self._encoder_out_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions),
        )

        self._init_weights()

    def forward(self, obs, hidden=None):
        """
        obs: (batch, C, H, W)
        Returns: q_values (batch, n_actions), None
        """
        x = self.encode(obs)
        return self.head(x), None

    def init_hidden(self, batch_size=1, device=None):
        return None