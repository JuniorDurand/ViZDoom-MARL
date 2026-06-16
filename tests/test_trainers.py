"""
Test script for PPOTrainer.
Run: PYTHONPATH=. python tests/test_trainers.py
"""

import sys
import time
from pathlib import Path
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_ppo_trainer():
    """Test PPO trainer short run."""
    from src.trainers.ppo_trainer import PPOTrainer, PPOConfig

    print("=" * 50)
    print("Test: PPO Trainer")
    print("=" * 50)

    config = PPOConfig(
        scenario="deathmatch",
        episode_minutes=1.0,
        rollout_length=64,
        rollouts_per_update=2,
        ppo_epochs=2,
        minibatch_rollouts=1,
        total_updates=3,
        log_every=1,
        save_every=2,
        eval_every=3,
        record_every=3,
        save_dir="/tmp/test_ppo",
        use_wandb=False,
    )

    trainer = PPOTrainer(config)
    trainer.train()
    trainer.close()

    assert (Path("/tmp/test_ppo/checkpoints/checkpoint_final.pt")).is_file()
    assert (Path("/tmp/test_ppo/metrics.json")).is_file()

    # verify metrics were tracked
    import json
    with open("/tmp/test_ppo/metrics.json") as f:
        metrics = json.load(f)
    assert len(metrics) == 3
    assert "train/policy_loss" in metrics[0]
    assert "system/steps_per_sec" in metrics[0]
    print(f"  Metrics entries: {len(metrics)}")
    print(f"  Keys: {sorted(metrics[-1].keys())}")

    # verify checkpoint loads
    from src.networks.actor_critic import RecurrentActorCritic
    from src.networks.agent import Agent

    ac = RecurrentActorCritic(in_channels=12, n_actions=19)
    agent = Agent(ac)
    agent.load("/tmp/test_ppo/checkpoints/checkpoint_final.pt")
    print("  Checkpoint load: OK")

    print("PASSED\n")


if __name__ == "__main__":
    test_ppo_trainer()
    print("=" * 50)
    print("All trainer tests passed!")
    print("=" * 50)