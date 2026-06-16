"""
On-policy rollout buffer for PPO/IPPO/MAPPO.

Unlike replay buffers, rollouts are used once and discarded.
Workers collect fixed-length rollouts and send them to the
learner via multiprocessing Queue.

Supports:
  - Recurrent policies (stores initial hidden state)
  - Centralized critic (stores global observations for MAPPO)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RolloutConfig:
    """Rollout buffer configuration."""
    length:      int   = 128
    obs_shape:   tuple = (12, 84, 84)
    n_actions:   int   = 19
    hidden_size: int   = 256
    gamma:       float = 0.99
    gae_lambda:  float = 0.95
    # MAPPO: shape of global observation for centralized critic
    global_obs_shape: Optional[tuple] = None


class RolloutBuffer:
    """
    Fixed-length rollout storage for a single agent.

    Stores transitions as numpy arrays (serializable for multiprocessing).
    After collection, compute_gae() calculates advantages and returns.

    Usage:
        buf = RolloutBuffer(config)
        for step in range(config.length):
            buf.store(step, obs, action, logprob, value, reward, done)
        buf.finish(last_value, last_done)

        # send buf.to_dict() via Queue to learner
        # learner calls RolloutBuffer.from_dict() and compute_gae()
    """

    def __init__(self, config: Optional[RolloutConfig] = None, **kwargs):
        if config is None:
            config = RolloutConfig(**kwargs)
        self.cfg = config
        T = config.length

        self.obs        = np.zeros((T, *config.obs_shape), dtype=np.uint8)
        self.actions    = np.zeros(T, dtype=np.int64)
        self.logprobs   = np.zeros(T, dtype=np.float32)
        self.values     = np.zeros(T, dtype=np.float32)
        self.rewards    = np.zeros(T, dtype=np.float32)
        self.dones      = np.zeros(T, dtype=np.float32)

        # recurrent hidden state at rollout start
        self.h_init = np.zeros(config.hidden_size, dtype=np.float32)

        # MAPPO: global observation for centralized critic
        if config.global_obs_shape is not None:
            self.global_obs = np.zeros((T, *config.global_obs_shape), dtype=np.uint8)
        else:
            self.global_obs = None

        # filled by finish()
        self.last_value: float = 0.0
        self.last_done:  float = 0.0

        # filled by compute_gae()
        self.advantages: Optional[np.ndarray] = None
        self.returns:    Optional[np.ndarray] = None

        self._step = 0

    def store(self, obs: np.ndarray, action: int, logprob: float,
              value: float, reward: float, done: float,
              global_obs: Optional[np.ndarray] = None):
        """Store a single transition at current step."""
        t = self._step
        self.obs[t] = obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done

        if self.global_obs is not None and global_obs is not None:
            self.global_obs[t] = global_obs

        self._step += 1

    def set_hidden(self, h: np.ndarray):
        """Store the GRU hidden state at rollout start."""
        self.h_init = h.copy()

    def finish(self, last_value: float, last_done: float):
        """Mark rollout as complete with bootstrap value."""
        self.last_value = last_value
        self.last_done = last_done

    @property
    def is_full(self) -> bool:
        return self._step >= self.cfg.length

    def reset(self):
        """Reset step counter for next rollout (reuses arrays)."""
        self._step = 0
        self.advantages = None
        self.returns = None

    # -- GAE ----------------------------------------------------------------

    def compute_gae(self):
        """
        Compute Generalized Advantage Estimation.

        Sets self.advantages and self.returns.
        Call this on the learner side after receiving the rollout.
        """
        T = self.cfg.length
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = self.last_value
        next_nonterminal = 1.0 - self.last_done

        for t in reversed(range(T)):
            delta = (self.rewards[t]
                     + gamma * next_value * next_nonterminal
                     - self.values[t])
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae

            next_value = self.values[t]
            next_nonterminal = 1.0 - self.dones[t]

        self.advantages = advantages
        self.returns = advantages + self.values

    # -- serialization (for multiprocessing Queue) --------------------------

    def to_dict(self) -> dict:
        """Serialize to dict for sending via Queue."""
        d = {
            "obs":        self.obs.copy(),
            "actions":    self.actions.copy(),
            "logprobs":   self.logprobs.copy(),
            "values":     self.values.copy(),
            "rewards":    self.rewards.copy(),
            "dones":      self.dones.copy(),
            "h_init":     self.h_init.copy(),
            "last_value": self.last_value,
            "last_done":  self.last_done,
        }
        if self.global_obs is not None:
            d["global_obs"] = self.global_obs.copy()
        return d

    @classmethod
    def from_dict(cls, d: dict, config: RolloutConfig) -> "RolloutBuffer":
        """Deserialize from dict received via Queue."""
        buf = cls(config)
        buf.obs        = d["obs"]
        buf.actions    = d["actions"]
        buf.logprobs   = d["logprobs"]
        buf.values     = d["values"]
        buf.rewards    = d["rewards"]
        buf.dones      = d["dones"]
        buf.h_init     = d["h_init"]
        buf.last_value = d["last_value"]
        buf.last_done  = d["last_done"]
        buf._step      = config.length

        if "global_obs" in d:
            buf.global_obs = d["global_obs"]

        return buf


class RolloutBatch:
    """
    Batches multiple rollouts into tensors for PPO update.

    Usage:
        rollouts = [RolloutBuffer.from_dict(d, cfg) for d in queue_data]
        batch = RolloutBatch.from_rollouts(rollouts)
        # batch.obs: (B, T, *obs_shape)
        # batch.advantages: (B, T) normalized
    """

    def __init__(self):
        self.obs:        np.ndarray = None
        self.actions:    np.ndarray = None
        self.logprobs:   np.ndarray = None
        self.values:     np.ndarray = None
        self.rewards:    np.ndarray = None
        self.dones:      np.ndarray = None
        self.advantages: np.ndarray = None
        self.returns:    np.ndarray = None
        self.h_init:     np.ndarray = None
        self.global_obs: np.ndarray = None
        self.batch_size: int = 0
        self.seq_len:    int = 0

    @classmethod
    def from_rollouts(cls, rollouts: list["RolloutBuffer"],
                      normalize_advantages: bool = True) -> "RolloutBatch":
        """Stack rollouts into batch arrays and compute GAE."""
        batch = cls()
        batch.batch_size = len(rollouts)
        batch.seq_len = rollouts[0].cfg.length

        # compute GAE for each rollout
        for r in rollouts:
            if r.advantages is None:
                r.compute_gae()

        batch.obs        = np.stack([r.obs for r in rollouts])
        batch.actions    = np.stack([r.actions for r in rollouts])
        batch.logprobs   = np.stack([r.logprobs for r in rollouts])
        batch.values     = np.stack([r.values for r in rollouts])
        batch.rewards    = np.stack([r.rewards for r in rollouts])
        batch.dones      = np.stack([r.dones for r in rollouts])
        batch.advantages = np.stack([r.advantages for r in rollouts])
        batch.returns    = np.stack([r.returns for r in rollouts])
        batch.h_init     = np.stack([r.h_init for r in rollouts])

        if rollouts[0].global_obs is not None:
            batch.global_obs = np.stack([r.global_obs for r in rollouts])

        if normalize_advantages:
            adv = batch.advantages
            batch.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        return batch