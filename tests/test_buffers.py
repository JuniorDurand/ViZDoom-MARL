"""
Test script for Buffers.
Run: PYTHONPATH=. python tests/test_rollout_buffer.py
"""

import sys
from pathlib import Path
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rollout_buffer():
    from src.buffers.rollout_buffer import RolloutBuffer, RolloutBatch, RolloutConfig

    print("=" * 50)
    print("Test: Rollout buffer")
    print("=" * 50)

    config = RolloutConfig(
        length=64,
        obs_shape=(12, 84, 84),
        n_actions=19,
        hidden_size=256,
    )

    buf = RolloutBuffer(config)
    buf.set_hidden(np.random.randn(256).astype(np.float32))

    for step in range(config.length):
        obs = np.random.randint(0, 255, config.obs_shape, dtype=np.uint8)
        action = np.random.randint(0, config.n_actions)
        buf.store(obs, action, float(np.random.randn()), float(np.random.randn()),
                  float(np.random.randn()) * 0.1, float(np.random.random() < 0.01))

    assert buf.is_full
    buf.finish(last_value=0.5, last_done=0.0)
    print(f"  Collection: OK (steps={config.length})")

    buf.compute_gae()
    assert buf.advantages is not None
    assert buf.returns is not None
    assert buf.advantages.shape == (config.length,)
    print(f"  GAE: OK (adv mean={buf.advantages.mean():.4f}, std={buf.advantages.std():.4f})")

    d = buf.to_dict()
    buf2 = RolloutBuffer.from_dict(d, config)
    assert np.array_equal(buf.obs, buf2.obs)
    assert np.array_equal(buf.actions, buf2.actions)
    print(f"  Serialization: OK")

    rollouts = []
    for _ in range(8):
        b = RolloutBuffer(config)
        b.set_hidden(np.random.randn(256).astype(np.float32))
        for step in range(config.length):
            b.store(
                np.random.randint(0, 255, config.obs_shape, dtype=np.uint8),
                np.random.randint(0, config.n_actions),
                float(np.random.randn()),
                float(np.random.randn()),
                float(np.random.randn()) * 0.1,
                float(np.random.random() < 0.01),
            )
        b.finish(last_value=0.0, last_done=1.0)
        rollouts.append(b)

    batch = RolloutBatch.from_rollouts(rollouts)
    print(f"  Batch shapes:")
    print(f"    obs:        {batch.obs.shape}")
    print(f"    actions:    {batch.actions.shape}")
    print(f"    advantages: {batch.advantages.shape}")
    print(f"    returns:    {batch.returns.shape}")
    print(f"    h_init:     {batch.h_init.shape}")

    assert batch.obs.shape == (8, 64, 12, 84, 84)
    assert batch.actions.shape == (8, 64)
    assert batch.advantages.shape == (8, 64)
    assert abs(batch.advantages.mean()) < 0.1
    print(f"  Batch: OK (normalized adv mean={batch.advantages.mean():.6f})")

    print("\n  Testing MAPPO global obs...")
    config_mappo = RolloutConfig(
        length=64,
        obs_shape=(12, 84, 84),
        global_obs_shape=(24, 84, 84),
    )
    buf_mappo = RolloutBuffer(config_mappo)
    for step in range(config_mappo.length):
        buf_mappo.store(
            np.random.randint(0, 255, config_mappo.obs_shape, dtype=np.uint8),
            np.random.randint(0, 19),
            float(np.random.randn()),
            float(np.random.randn()),
            float(np.random.randn()) * 0.1,
            0.0,
            global_obs=np.random.randint(0, 255, config_mappo.global_obs_shape, dtype=np.uint8),
        )
    buf_mappo.finish(0.0, 1.0)
    d = buf_mappo.to_dict()
    assert "global_obs" in d
    assert d["global_obs"].shape == (64, 24, 84, 84)
    print(f"  MAPPO global obs: OK ({d['global_obs'].shape})")

    print("PASSED\n")



if __name__ == "__main__":
    test_rollout_buffer()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)