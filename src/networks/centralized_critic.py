"""
Centralized critic for MAPPO.

Receives concatenated observations from all agents and outputs
a single value estimate. Used only during training (CTDE paradigm).

Architecture mirrors the actor's encoder for fair comparison,
but input has n_agents * in_channels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralizedCritic(nn.Module):
    """
    Shared value function for MAPPO.

    One instance is shared across all agents. Each agent's value
    is estimated from the same global observation.

    Input:  (batch, n_agents * in_channels, H, W)
    Output: (batch,) value
    """

    def __init__(self, n_agents: int, in_channels: int,
                 hidden_size: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.global_channels = n_agents * in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.global_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self._encoder_out_size = self._compute_encoder_output()
        self.fc = nn.Linear(self._encoder_out_size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _compute_encoder_output(self) -> int:
        dummy = torch.zeros(1, self.global_channels, 84, 84)
        with torch.no_grad():
            out = self.encoder(dummy)
        return out.view(1, -1).shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, global_obs: torch.Tensor,
                hidden: torch.Tensor = None):
        """
        Args:
            global_obs: (batch, n_agents * in_channels, H, W)
            hidden: (batch, hidden_size)

        Returns:
            value: (batch,)
            hidden: (batch, hidden_size)
        """
        x = global_obs.float() / 255.0 if global_obs.dtype == torch.uint8 else global_obs
        x = self.encoder(x).flatten(1)
        x = F.relu(self.fc(x), inplace=True)

        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        hidden = self.gru_cell(x, hidden)
        value = self.value_head(hidden).squeeze(-1)

        return value, hidden

    def init_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)