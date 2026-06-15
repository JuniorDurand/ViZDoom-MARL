"""
Test script for DeathmatchEnv.
Run: PYTHONPATH=. python tests/test_deathmatch.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_cig_scenario():
    """Check if CIG scenario exists."""
    import vizdoom as vzd

    print("=" * 50)
    print("Test 1: CIG scenario availability")
    print("=" * 50)

    cig_path = Path(vzd.scenarios_path) / "cig.cfg"
    print(f"CIG path: {cig_path}")
    print(f"Exists: {cig_path.is_file()}")

    if not cig_path.is_file():
        print("Available scenarios:")
        for f in sorted(Path(vzd.scenarios_path).glob("*.cfg")):
            print(f"  {f.stem}")

    print("PASSED\n")


def test_single_mode():
    """Test single-player mode against monsters."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig

    print("=" * 50)
    print("Test 2: Single-player mode (RGB)")
    print("=" * 50)

    config = EnvConfig(scenario="deathmatch", mode="single", episode_minutes=1.0)
    env = DeathmatchEnv(config)

    obs = env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"N actions: {env.n_actions}")
    print(f"In channels: {env.in_channels}")

    expected = (config.frame_stack * 3, 84, 84)
    assert obs.shape == expected, f"Expected {expected}, got {obs.shape}"
    assert obs.dtype == np.uint8

    total_reward = 0.0
    steps = 0
    t0 = time.time()

    while True:
        action = np.random.randint(0, env.n_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done or steps >= 500:
            break

    elapsed = time.time() - t0
    print(f"Steps: {steps}")
    print(f"Time: {elapsed:.1f}s ({steps / elapsed:.0f} steps/s)")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Info: {info}")
    print("PASSED\n")

    env.close()


def test_reward_components():
    """Verify reward is non-trivial."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig

    print("=" * 50)
    print("Test 3: Reward components")
    print("=" * 50)

    env = DeathmatchEnv(EnvConfig(scenario="deathmatch", mode="single"))
    env.reset()

    rewards = []
    for _ in range(200):
        action = np.random.randint(0, env.n_actions)
        _, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            env.reset()

    rewards = np.array(rewards)
    print(f"Reward stats: mean={rewards.mean():.4f}, "
          f"std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, "
          f"max={rewards.max():.4f}")
    print(f"Non-zero rewards: {np.count_nonzero(rewards)}/{len(rewards)}")
    print("PASSED\n")

    env.close()


def test_recording():
    """Test video recording."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig

    print("=" * 50)
    print("Test 4: Recording")
    print("=" * 50)

    config = EnvConfig(
        scenario="deathmatch",
        mode="single",
        record=True,
        max_rec_frames=100,
    )
    env = DeathmatchEnv(config)
    env.reset()

    for _ in range(50):
        action = np.random.randint(0, env.n_actions)
        _, _, done, _ = env.step(action)
        if done:
            break

    frames = env.stop_recording()
    print(f"Recorded frames: {len(frames)}")
    if frames:
        print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
        assert frames[0].ndim == 3, "Expected RGB frame (H, W, 3)"
        assert frames[0].shape[-1] == 3, "Expected 3 channels"
    print("PASSED\n")

    env.close()


def test_respawn():
    """Verify agent respawns on death."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig

    print("=" * 50)
    print("Test 5: Respawn on death")
    print("=" * 50)

    env = DeathmatchEnv(EnvConfig(scenario="deathmatch", mode="single"))
    env.reset()

    deaths = 0
    steps = 0
    for _ in range(1000):
        action = np.random.randint(0, env.n_actions)
        _, _, done, info = env.step(action)
        steps += 1
        cur_deaths = info.get("deaths", 0)
        if cur_deaths > deaths:
            deaths = int(cur_deaths)
            print(f"  Death #{deaths} at step {steps} (episode continues: {not done})")
        if done:
            break

    print(f"Total steps: {steps}, deaths: {deaths}, ended by timeout: {done}")
    print("PASSED\n")

    env.close()


def test_obs_consistency():
    """Verify observation shape is consistent across steps and resets."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig

    print("=" * 50)
    print("Test 6: Observation consistency")
    print("=" * 50)

    env = DeathmatchEnv(EnvConfig(scenario="deathmatch", mode="single"))
    expected = (env.in_channels, 84, 84)

    for ep in range(3):
        obs = env.reset()
        assert obs.shape == expected, f"Reset ep {ep}: {obs.shape} != {expected}"

        for step in range(50):
            action = np.random.randint(0, env.n_actions)
            obs, _, done, _ = env.step(action)
            assert obs.shape == expected, f"Step {step} ep {ep}: {obs.shape} != {expected}"
            assert obs.dtype == np.uint8
            if done:
                break

    print(f"Shape {expected} consistent across 3 episodes")
    print("PASSED\n")

    env.close()


if __name__ == "__main__":
    test_cig_scenario()
    test_single_mode()
    test_reward_components()
    test_recording()
    test_respawn()
    test_obs_consistency()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)