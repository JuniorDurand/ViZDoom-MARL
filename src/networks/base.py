"""
Base class for all agent networks.

The trainer uses `has_memory` to decide:
  - True  → EpisodeReplayBuffer, sequential input (batch, seq_len, C, H, W)
  - False → TransitionReplayBuffer, single input (batch, C, H, W)
"""

import math

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Interface that all agent networks implement.

    Subclasses must set `has_memory` and implement `forward` and `init_hidden`.
    The CNN encoder is shared across all architectures for fair comparison.
    """

    has_memory: bool = False

    def __init__(self, in_channels: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self._encoder_out_size = self._compute_encoder_output(in_channels)

    def _compute_encoder_output(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            out = self.encoder(dummy)
        return out.view(1, -1).shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """(batch, C, H, W) uint8 or float → (batch, encoder_dim) float."""
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs
        return self.encoder(x).flatten(1)

    def forward(self, obs, hidden=None):
        raise NotImplementedError

    def init_hidden(self, batch_size: int = 1, device=None):
        return None