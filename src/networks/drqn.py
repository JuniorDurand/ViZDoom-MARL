"""
DRQN: CNN encoder → GRU → Q-values.
Temporal memory via GRU for partially observable environments.
Processes sequences for training with EpisodeReplayBuffer.
"""

import torch
import torch.nn as nn

from src.networks.base import BaseNetwork


class DRQNetwork(BaseNetwork):

    has_memory = True

    def __init__(self, in_channels: int, n_actions: int, hidden_size: int = 256):
        super().__init__(in_channels, n_actions, hidden_size)

        self.fc = nn.Linear(self._encoder_out_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.q_head = nn.Linear(hidden_size, n_actions)

        self._init_weights()

    def forward(self, obs, hidden=None):
        """
        obs: (batch, seq_len, C, H, W)
        hidden: (1, batch, hidden_size)
        Returns: q_values (batch, seq_len, n_actions), hidden
        """
        B, T, C, H, W = obs.shape

        x = self.encode(obs.reshape(B * T, C, H, W))
        x = torch.relu(self.fc(x))
        x = x.reshape(B, T, -1)

        if hidden is None:
            x, hidden = self.gru(x)
        else:
            x, hidden = self.gru(x, hidden)

        return self.q_head(x), hidden

    def init_hidden(self, batch_size=1, device=None):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)