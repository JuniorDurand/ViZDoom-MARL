"""
Experiment 1: PPO pretraining on deathmatch (v2 - aggressive reward).

Run: PYTHONPATH=. python experiments/exp1_ppo_pretrain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainers.ppo_trainer import PPOTrainer, PPOConfig
from src.envs.deathmatch import RewardConfig


def main():
    config = PPOConfig(
        # ---- environment ----
        scenario="deathmatch",
        episode_minutes=4.0,
        doom_skill=3,

        # ---- reward shaping ----
        reward=RewardConfig(
            # reward += frag * (CUR.FRAGCOUNT - PREV.FRAGCOUNT)
            frag=5.0,
            # reward += damage * max(0, CUR.DAMAGECOUNT - PREV.DAMAGECOUNT)
            damage=0.5,
            # reward += damage_taken * max(0, CUR.DAMAGE_TAKEN - PREV.DAMAGE_TAKEN)  [negative value → penalty]
            damage_taken=-0.01,
            # reward += hit * max(0, CUR.HITCOUNT - PREV.HITCOUNT)
            hit=0.5,
            # reward += item * max(0, CUR.ITEMCOUNT - PREV.ITEMCOUNT)
            item=0.005,
            # reward += health * (CUR.HEALTH - PREV.HEALTH)  [only when delta > 0]
            health=0.0005,
            # reward += ammo * max(0, CUR.SELECTED_WEAPON_AMMO - PREV.SELECTED_WEAPON_AMMO)
            ammo=0.002,
            # reward -= death  [when player is dead]
            death=1.0,
            # reward += living  [every step]
            living=0.0001,
            # reward -= miss  [when ATTACK action fired but HITCOUNT unchanged]
            miss=0.05,
            # reward = clip(reward, clip_min, clip_max)
            clip_min=-3.0,
            clip_max=5.0,
        ),

        # ---- rollout ----
        rollout_length=128,
        rollouts_per_update=4,

        # ---- PPO ----
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_rollouts=2,
        value_clip=0.2,

        # ---- network ----
        hidden_size=256,

        # ---- training ----
        total_updates=2000,
        log_every=10,
        save_every=200,
        eval_every=100,
        eval_episodes=3,
        record_every=200,
        seed=42,

        # ---- paths ----
        save_dir="./results/exp1_ppo_pretrain",
        resume=None,

        # ---- logging ----
        use_wandb=True,
        wandb_project="doom-marl",
        wandb_run_name="ppo-pretrain-v3-aggressive",
        wandb_tags=["ppo", "pretrain", "deathmatch", "v3", "aggressive-reward"],
    )

    trainer = PPOTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted, saving...")
        trainer._save_checkpoint("checkpoint_interrupted")
        trainer._save_metrics()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()