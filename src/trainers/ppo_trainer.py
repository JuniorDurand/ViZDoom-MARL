"""
PPO trainer for single-agent pretraining.

Trains a recurrent actor-critic against built-in monsters in
single-player mode. The resulting weights can be transferred
to IPPO/MAPPO for multiplayer fine-tuning.

Features:
    - W&B logging (optional) with game metrics and training stats
    - Periodic video recording of agent gameplay
    - Checkpoint saving for transfer learning
    - Evaluation episodes with greedy policy
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.envs.deathmatch import DeathmatchEnv, EnvConfig, RewardConfig
from src.networks.actor_critic import RecurrentActorCritic
from src.networks.agent import Agent
from src.buffers.rollout_buffer import RolloutBuffer, RolloutBatch, RolloutConfig

from dotenv import load_dotenv
load_dotenv()


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # environment
    scenario:        str   = "deathmatch"
    episode_minutes: float = 4.0
    doom_skill:      int   = 4
    reward:          RewardConfig = None  # None = defaults

    # rollout
    rollout_length:  int   = 128
    rollouts_per_update: int = 4

    # PPO
    lr:              float = 2.5e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip:            float = 0.1
    vf_coef:         float = 0.5
    ent_coef:        float = 0.005
    max_grad_norm:   float = 0.5
    ppo_epochs:      int   = 4
    minibatch_rollouts: int = 2
    value_clip:      float = 0.2

    # training
    total_updates:   int   = 1000
    log_every:       int   = 5
    save_every:      int   = 100
    eval_every:      int   = 50
    eval_episodes:   int   = 3
    record_every:    int   = 100

    # paths
    save_dir:        str   = "./results/ppo_pretrain"
    resume:          Optional[str] = None

    # network
    hidden_size:     int   = 256

    # reproducibility
    seed:            Optional[int] = 42

    # logging
    use_wandb:       bool  = False
    wandb_project:   str   = "doom-marl"
    wandb_run_name:  Optional[str] = None
    wandb_tags:      list  = None


class PPOTrainer:
    """
    Single-agent PPO trainer for pretraining against bots.

    Collects rollouts in single-player mode (fast, no networking),
    performs PPO updates, and saves checkpoints for transfer learning.
    """

    def __init__(self, config: Optional[PPOConfig] = None, **kwargs):
        if config is None:
            config = PPOConfig(**kwargs)
        self.cfg = config

        # reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # dirs
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "videos").mkdir(exist_ok=True)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # reward config
        reward_cfg = config.reward or RewardConfig()

        # environment
        self.env = DeathmatchEnv(EnvConfig(
            scenario=config.scenario,
            mode="single",
            episode_minutes=config.episode_minutes,
            doom_skill=config.doom_skill,
            reward=reward_cfg,
        ))

        # agent
        ac = RecurrentActorCritic(
            in_channels=self.env.in_channels,
            n_actions=self.env.n_actions,
            hidden_size=config.hidden_size,
        )
        self.agent = Agent(ac, device=self.device)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=config.lr,
            eps=1e-5,
            weight_decay=1e-6,
        )

        # rollout config
        self.rollout_cfg = RolloutConfig(
            length=config.rollout_length,
            obs_shape=self.env.obs_shape,
            n_actions=self.env.n_actions,
            hidden_size=config.hidden_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # tracking
        self.updates = 0
        self.total_steps = 0
        self.episode_count = 0
        self.metrics_history = []

        # persistent state across rollouts
        self._obs = self.env.reset()
        self._actor_h = self.agent.init_hidden(1)
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_rewards = []
        self._episode_infos = []

        # wandb
        self._wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        # resume
        if config.resume and Path(config.resume).is_file():
            self._load_checkpoint(config.resume)

    # -- wandb ----------------------------------------------------------------

    def _init_wandb(self):
        try:
            import os
            import wandb

            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
            else:
                print("WANDB_API_KEY not found in .env, trying default auth")

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name or f"ppo_{self.cfg.scenario}",
                tags=self.cfg.wandb_tags or ["ppo", "pretrain", self.cfg.scenario],
                config=asdict(self.cfg),
                save_code=True,
            )
            print(f"W&B run: {wandb.run.url}")
        except ImportError:
            print("wandb not installed, logging disabled")
            self.cfg.use_wandb = False

    def _log_wandb(self, metrics: dict, step: int):
        if self._wandb_run is None:
            return
        import wandb
        wandb.log(metrics, step=step)

    def _log_wandb_video(self, path: str, caption: str):
        if self._wandb_run is None:
            return
        import wandb
        try:
            wandb.log({"gameplay": wandb.Video(path, caption=caption, fps=8)})
        except Exception:
            pass

    def _log_wandb_artifact(self, path: str, name: str, artifact_type: str):
        if self._wandb_run is None:
            return
        import wandb
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        self._wandb_run.log_artifact(artifact)

    # -- rollout collection ---------------------------------------------------

    def _collect_rollout(self) -> RolloutBuffer:
        buf = RolloutBuffer(self.rollout_cfg)
        buf.set_hidden(self._actor_h.squeeze(0).cpu().numpy())

        self.agent.eval()

        for _ in range(self.cfg.rollout_length):
            action, logprob, value, self._actor_h, _ = self.agent.act(
                self._obs, self._actor_h)

            next_obs, reward, done, info = self.env.step(action)
            buf.store(self._obs, action, logprob, value, reward, float(done))

            self._obs = next_obs
            self._episode_reward += reward
            self._episode_steps += 1
            self.total_steps += 1

            if done:
                self._episode_rewards.append(self._episode_reward)
                self._episode_infos.append(info)
                self.episode_count += 1

                self._obs = self.env.reset()
                self._actor_h = self.agent.init_hidden(1)
                self._episode_reward = 0.0
                self._episode_steps = 0

        last_value = self.agent.get_value(self._obs, self._actor_h)
        buf.finish(last_value, 0.0)

        return buf

    # -- PPO update -----------------------------------------------------------

    def _update(self, rollouts: list) -> dict:
        self.agent.train()
        batch = RolloutBatch.from_rollouts(rollouts)

        B = batch.batch_size
        cfg = self.cfg

        obs = torch.from_numpy(batch.obs).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        old_logprobs = torch.from_numpy(batch.logprobs).to(self.device)
        advantages = torch.from_numpy(batch.advantages).to(self.device)
        returns = torch.from_numpy(batch.returns).to(self.device)
        old_values = torch.from_numpy(batch.values).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        h_init = torch.from_numpy(batch.h_init).to(self.device)

        stats = {
            "train/policy_loss": 0.0,
            "train/value_loss": 0.0,
            "train/entropy": 0.0,
            "train/approx_kl": 0.0,
            "train/clip_fraction": 0.0,
        }
        n_minibatches = 0

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=self.device)

            for start in range(0, B, cfg.minibatch_rollouts):
                idx = perm[start:start + cfg.minibatch_rollouts]
                if idx.numel() == 0:
                    continue

                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_old_lp = old_logprobs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_old_v = old_values[idx]
                mb_dones = dones[idx]
                mb_h = h_init[idx]

                new_logprobs, values, entropy = self.agent.evaluate_actions(
                    mb_obs, mb_act, mb_h, mb_dones)

                ratio = torch.exp(new_logprobs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_v + torch.clamp(
                    values - mb_old_v, -cfg.value_clip, cfg.value_clip)
                vloss1 = (values - mb_ret).pow(2)
                vloss2 = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(vloss1, vloss2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_logprobs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean().item()

                stats["train/policy_loss"] += policy_loss.item()
                stats["train/value_loss"] += value_loss.item()
                stats["train/entropy"] += entropy.item()
                stats["train/approx_kl"] += approx_kl
                stats["train/clip_fraction"] += clip_frac
                n_minibatches += 1

        for k in stats:
            stats[k] /= max(n_minibatches, 1)

        self.updates += 1
        return stats
    
    def _make_env_config(self, record=False) -> EnvConfig:
        """Shared env config for eval and recording."""
        return EnvConfig(
            scenario=self.cfg.scenario,
            mode="single",
            episode_minutes=self.cfg.episode_minutes,
            doom_skill=self.cfg.doom_skill,
            reward=self.cfg.reward or RewardConfig(),
            record=record,
            max_rec_frames=4000 if record else 0,
        )

    # -- evaluation -----------------------------------------------------------

    def _evaluate(self, n_episodes: int) -> dict:
        self.agent.eval()
        eval_env = DeathmatchEnv(self._make_env_config())

        all_rewards = []
        all_infos = []

        for _ in range(n_episodes):
            obs = eval_env.reset()
            h = self.agent.init_hidden(1)
            done = False
            ep_reward = 0.0

            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _, h = self.agent.actor_critic(obs_t, h)
                    dist = Categorical(logits=logits)
                    if np.random.random() < 0.04:
                        action = dist.sample().item()
                    else:
                        action = logits.argmax(dim=1).item()

                obs, reward, done, info = eval_env.step(action)
                ep_reward += reward

            all_rewards.append(ep_reward)
            all_infos.append(info)

        eval_env.close()

        return {
            "eval/reward_mean":   np.mean(all_rewards),
            "eval/reward_std":    np.std(all_rewards),
            "eval/frags_mean":    np.mean([i["frags"] for i in all_infos]),
            "eval/kills_mean":    np.mean([i["kills"] for i in all_infos]),
            "eval/total_kills_mean": np.mean([i["total_kills"] for i in all_infos]),
            "eval/deaths_mean":   np.mean([i["deaths"] for i in all_infos]),
            "eval/kd_ratio":      sum(i["total_kills"] for i in all_infos) / max(sum(i["deaths"] for i in all_infos), 1),
            "eval/damage_dealt":  np.mean([i["damage_dealt"] for i in all_infos]),
            "eval/damage_taken":  np.mean([i["damage_taken"] for i in all_infos]),
            "eval/damage_ratio":  np.mean([i["damage_ratio"] for i in all_infos]),
            "eval/hit_rate":      np.mean([i["hit_rate"] for i in all_infos]),
        }

    # -- recording ------------------------------------------------------------

    def _record_episode(self) -> Optional[str]:
        """Record one episode and save as video."""
        rec_env = DeathmatchEnv(self._make_env_config(record=True))

        obs = rec_env.reset()
        h = self.agent.init_hidden(1)
        done = False

        self.agent.eval()
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _, h = self.agent.actor_critic(obs_t, h)
                dist = Categorical(logits=logits)
                if np.random.random() < 0.04:
                    action = dist.sample().item()
                else:
                    action = logits.argmax(dim=1).item()
            obs, _, done, info = rec_env.step(action)

        frames = rec_env.stop_recording()
        rec_env.close()

        if not frames:
            return None

        path = str(self.save_dir / "videos" / f"update_{self.updates:05d}.mp4")
        try:
            import imageio
            clean = []
            for f in frames:
                f = np.asarray(f, dtype=np.uint8)
                if f.ndim == 3 and f.shape[-1] == 3:
                    clean.append(f)
            if clean:
                imageio.mimsave(path, clean, fps=8, codec="libx264",
                                quality=9, macro_block_size=1,
                                output_params=["-pix_fmt", "yuv420p"])
                return path
        except ImportError:
            print("  [video] imageio not installed, skipping")
        except Exception as e:
            print(f"  [video] save failed: {e}")
        return None

    # -- checkpoint -----------------------------------------------------------

    def _save_checkpoint(self, name: str = "checkpoint") -> Path:
        path = self.save_dir / "checkpoints" / f"{name}.pt"
        torch.save({
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }, path)
        return path

    def _save_metrics(self):
        path = self.save_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        self.updates = ckpt.get("updates", 0)
        self.total_steps = ckpt.get("total_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        print(f"Resumed from {path} (update {self.updates})")

    # -- main loop ------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        print(f"Device: {self.device}")
        print(f"Scenario: {cfg.scenario}")
        print(f"Rollout: {cfg.rollout_length} steps x {cfg.rollouts_per_update} per update")
        print(f"PPO: {cfg.ppo_epochs} epochs, minibatch {cfg.minibatch_rollouts} rollouts")
        print(f"Training for {cfg.total_updates} updates")
        print(f"Save dir: {self.save_dir}")
        if self._wandb_run:
            print(f"W&B: {self._wandb_run.url}")
        print()

        t_start = time.time()

        while self.updates < cfg.total_updates:
            # collect
            rollouts = []
            for _ in range(cfg.rollouts_per_update):
                rollouts.append(self._collect_rollout())

            # update
            stats = self._update(rollouts)

            # episode metrics
            if self._episode_infos:
                recent = self._episode_infos[-10:]
                stats["game/ep_reward_mean"]   = np.mean(self._episode_rewards[-10:])
                stats["game/frags_mean"]       = np.mean([i["frags"] for i in recent])
                stats["game/kills_mean"]       = np.mean([i["kills"] for i in recent])
                stats["game/total_kills_mean"] = np.mean([i["total_kills"] for i in recent])
                stats["game/deaths_mean"]      = np.mean([i["deaths"] for i in recent])
                stats["game/damage_dealt"]     = np.mean([i["damage_dealt"] for i in recent])
                stats["game/damage_taken"]     = np.mean([i["damage_taken"] for i in recent])
                stats["game/damage_ratio"]     = np.mean([i["damage_ratio"] for i in recent])
                stats["game/hits"]             = np.mean([i["hits"] for i in recent])
                stats["game/hit_rate"]         = np.mean([i["hit_rate"] for i in recent])
                stats["game/items"]            = np.mean([i["items"] for i in recent])
                stats["game/episode_steps"]    = np.mean([i["episode_steps"] for i in recent])
                stats["game/survival_steps"]   = np.mean([i["survival_steps"] for i in recent])
                total_kills = sum(i["total_kills"] for i in recent)
                total_deaths = sum(i["deaths"] for i in recent)
                stats["game/kd_ratio"] = total_kills / max(total_deaths, 1)

            # system
            elapsed = time.time() - t_start
            stats["system/steps_per_sec"] = self.total_steps / max(elapsed, 1)
            stats["system/total_steps"] = self.total_steps
            stats["system/episodes"] = self.episode_count
            stats["system/updates"] = self.updates

            self.metrics_history.append(stats)

            # wandb
            self._log_wandb(stats, step=self.total_steps)

            # console log
            if self.updates % cfg.log_every == 0:
                sps = stats["system/steps_per_sec"]
                parts = [
                    f"upd={self.updates:>5}",
                    f"steps={self.total_steps:>9}",
                    f"sps={sps:6.0f}",
                    f"pi={stats['train/policy_loss']:+.4f}",
                    f"vf={stats['train/value_loss']:.4f}",
                    f"ent={stats['train/entropy']:.3f}",
                    f"kl={stats['train/approx_kl']:.4f}",
                ]
                if "game/ep_reward_mean" in stats:
                    parts.append(f"rew={stats['game/ep_reward_mean']:.2f}")
                if "game/frags_mean" in stats:
                    parts.append(f"kills={stats['game/total_kills_mean']:.1f}")
                if "game/kd_ratio" in stats:
                    parts.append(f"K/D={stats['game/kd_ratio']:.2f}")
                if "game/hit_rate" in stats:
                    parts.append(f"hit={stats['game/hit_rate']:.3f}")
                print(" | ".join(parts), flush=True)

            # save
            if self.updates % cfg.save_every == 0:
                path = self._save_checkpoint("checkpoint_latest")
                self._save_checkpoint(f"checkpoint_{self.updates:05d}")
                self._save_metrics()
                self._log_wandb_artifact(str(path), f"model-{self.updates}", "model")
                print(f"  [save] {path}", flush=True)

            # eval
            if self.updates % cfg.eval_every == 0:
                eval_stats = self._evaluate(cfg.eval_episodes)
                self._log_wandb(eval_stats, step=self.total_steps)
                print(
                    f"  [eval] rew={eval_stats['eval/reward_mean']:.2f} "
                    f"frags={eval_stats['eval/frags_mean']:.1f} "
                    f"deaths={eval_stats['eval/deaths_mean']:.1f} "
                    f"K/D={eval_stats['eval/kd_ratio']:.2f} "
                    f"hit={eval_stats['eval/hit_rate']:.3f}",
                    flush=True,
                )

            # record video
            if self.updates % cfg.record_every == 0:
                video_path = self._record_episode()
                if video_path:
                    caption = f"update {self.updates}"
                    self._log_wandb_video(video_path, caption)
                    print(f"  [video] {video_path}", flush=True)

        # final
        self._save_checkpoint("checkpoint_final")
        self._save_metrics()
        self._log_wandb_artifact(
            str(self.save_dir / "checkpoints" / "checkpoint_final.pt"),
            "model-final", "model",
        )

        if self._wandb_run:
            self._wandb_run.finish()

        print(f"\nTraining complete. {self.updates} updates, {self.total_steps} steps.")

    def close(self):
        self.env.close()
        if self._wandb_run:
            self._wandb_run.finish()