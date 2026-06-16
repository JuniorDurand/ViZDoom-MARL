"""
Agent wrapper that unifies actor-critic interface across PPO/IPPO/MAPPO.

Encapsulates:
  - Actor policy (always local observations)
  - Value function (local critic or centralized critic)
  - Hidden state management

The trainer interacts only with this interface, never touching
the underlying networks directly. This keeps the training loop
identical for PPO, IPPO, and MAPPO.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.networks.actor_critic import RecurrentActorCritic
from src.networks.centralized_critic import CentralizedCritic


class Agent:
    """
    Unified agent interface.

    Modes:
      - PPO/IPPO: actor_critic handles both policy and value
      - MAPPO: actor_critic handles policy, centralized_critic handles value

    The caller doesn't need to know which mode is active.
    """

    def __init__(self, actor_critic: RecurrentActorCritic,
                 centralized_critic: Optional[CentralizedCritic] = None,
                 device: torch.device = None):

        self.actor_critic = actor_critic
        self.centralized_critic = centralized_critic
        self.device = device or torch.device("cpu")
        self.is_mappo = centralized_critic is not None

        self.actor_critic.to(self.device)
        if self.centralized_critic is not None:
            self.centralized_critic.to(self.device)

    # -- properties -----------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.actor_critic.hidden_size

    @property
    def critic_hidden_size(self) -> int:
        if self.is_mappo:
            return self.centralized_critic.hidden_size
        return self.actor_critic.hidden_size

    # -- hidden state management ----------------------------------------------

    def init_hidden(self, batch_size: int = 1):
        """Initialize actor hidden state."""
        return self.actor_critic.init_hidden(batch_size, self.device)

    def init_critic_hidden(self, batch_size: int = 1):
        """Initialize critic hidden state (separate for MAPPO)."""
        if self.is_mappo:
            return self.centralized_critic.init_hidden(batch_size, self.device)
        return self.actor_critic.init_hidden(batch_size, self.device)

    # -- rollout collection (one step at a time) ------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, actor_hidden: torch.Tensor,
            critic_hidden: Optional[torch.Tensor] = None,
            global_obs: Optional[np.ndarray] = None):
        """
        Select action for one step during rollout collection.

        Args:
            obs: local observation (in_channels, H, W) uint8
            actor_hidden: (1, hidden_size)
            critic_hidden: (1, hidden_size) only needed for MAPPO
            global_obs: (global_channels, H, W) uint8, only for MAPPO

        Returns:
            action: int
            logprob: float
            value: float
            actor_hidden: updated actor hidden
            critic_hidden: updated critic hidden (same as actor for PPO/IPPO)
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, local_value, actor_hidden = self.actor_critic(obs_t, actor_hidden)

        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        if self.is_mappo and global_obs is not None:
            global_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
            if critic_hidden is None:
                critic_hidden = self.init_critic_hidden(1)
            value, critic_hidden = self.centralized_critic(global_t, critic_hidden)
            value = value.item()
        else:
            value = local_value.item()
            critic_hidden = actor_hidden

        return (int(action.item()), float(logprob.item()), value,
                actor_hidden, critic_hidden)

    @torch.no_grad()
    def get_value(self, obs: np.ndarray, actor_hidden: torch.Tensor,
                  critic_hidden: Optional[torch.Tensor] = None,
                  global_obs: Optional[np.ndarray] = None) -> float:
        """
        Get value estimate for bootstrap at end of rollout.

        Returns:
            value: float
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        if self.is_mappo and global_obs is not None:
            global_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
            if critic_hidden is None:
                critic_hidden = self.init_critic_hidden(1)
            value, _ = self.centralized_critic(global_t, critic_hidden)
            return value.item()
        else:
            _, value, _ = self.actor_critic(obs_t, actor_hidden)
            return value.item()

    # -- PPO update (sequences) -----------------------------------------------

    def evaluate_actions(self, obs_seq: torch.Tensor,
                         actions_seq: torch.Tensor,
                         actor_h_init: torch.Tensor,
                         dones_seq: torch.Tensor,
                         critic_h_init: Optional[torch.Tensor] = None,
                         global_obs_seq: Optional[torch.Tensor] = None):
        """
        Re-evaluate actions for PPO loss computation.

        Args:
            obs_seq: (batch, T, C, H, W)
            actions_seq: (batch, T)
            actor_h_init: (batch, hidden_size)
            dones_seq: (batch, T)
            critic_h_init: (batch, hidden_size) for MAPPO
            global_obs_seq: (batch, T, global_C, H, W) for MAPPO

        Returns:
            new_logprobs: (batch, T)
            values: (batch, T)
            entropy: scalar
        """
        B, T = actions_seq.shape

        # actor forward: get logits for all timesteps
        logits_list = []
        local_values_list = []
        h = actor_h_init

        for t in range(T):
            if t > 0:
                mask = (1.0 - dones_seq[:, t - 1]).unsqueeze(-1)
                h = h * mask
            logits, value, h = self.actor_critic(obs_seq[:, t], h)
            logits_list.append(logits)
            local_values_list.append(value)

        logits_all = torch.stack(logits_list, dim=1)      # (B, T, n_actions)
        local_values = torch.stack(local_values_list, dim=1)  # (B, T)

        dist = Categorical(logits=logits_all)
        new_logprobs = dist.log_prob(actions_seq)
        entropy = dist.entropy().mean()

        # values: local or centralized
        if self.is_mappo and global_obs_seq is not None:
            critic_values_list = []
            ch = critic_h_init if critic_h_init is not None else \
                self.init_critic_hidden(B)

            for t in range(T):
                if t > 0:
                    mask = (1.0 - dones_seq[:, t - 1]).unsqueeze(-1)
                    ch = ch * mask
                v, ch = self.centralized_critic(global_obs_seq[:, t], ch)
                critic_values_list.append(v)

            values = torch.stack(critic_values_list, dim=1)  # (B, T)
        else:
            values = local_values

        return new_logprobs, values, entropy

    # -- save/load ------------------------------------------------------------

    def state_dict(self) -> dict:
        d = {"actor_critic": self.actor_critic.state_dict()}
        if self.is_mappo:
            d["centralized_critic"] = self.centralized_critic.state_dict()
        return d

    def load_state_dict(self, state: dict):
        self.actor_critic.load_state_dict(state["actor_critic"])
        if self.is_mappo and "centralized_critic" in state:
            self.centralized_critic.load_state_dict(state["centralized_critic"])

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        # handle trainer checkpoint format (wrapped in "agent" key)
        if "agent" in state:
            state = state["agent"]
        self.load_state_dict(state)

    def load_pretrained_actor(self, path: str):
        """
        Load only the actor weights from a pretrained checkpoint.
        Used for transfer learning: pretrain PPO → fine-tune IPPO/MAPPO.
        Critic is left randomly initialized.
        """
        state = torch.load(path, map_location=self.device)
        if "actor_critic" in state:
            self.actor_critic.load_state_dict(state["actor_critic"])
        else:
            self.actor_critic.load_state_dict(state)

    # -- parameters -----------------------------------------------------------

    def parameters(self):
        """All trainable parameters (actor + critic)."""
        params = list(self.actor_critic.parameters())
        if self.is_mappo:
            params += list(self.centralized_critic.parameters())
        return params

    def train(self):
        self.actor_critic.train()
        if self.is_mappo:
            self.centralized_critic.train()

    def eval(self):
        self.actor_critic.eval()
        if self.is_mappo:
            self.centralized_critic.eval()