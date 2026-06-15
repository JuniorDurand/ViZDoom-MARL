"""
Test script for networks.
Run: PYTHONPATH=. python tests/test_networks.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_dqn():
    from src.networks.dqn import DQNetwork

    print("=" * 50)
    print("Test 1: DQNetwork")
    print("=" * 50)

    net = DQNetwork(in_channels=12, n_actions=19)
    obs = torch.randint(0, 255, (8, 12, 84, 84), dtype=torch.uint8)

    q, h = net(obs)
    print(f"has_memory: {net.has_memory}")
    print(f"Q shape: {q.shape}")
    print(f"Hidden: {h}")
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    assert q.shape == (8, 19)
    assert h is None
    print("PASSED\n")


def test_drqn():
    from src.networks.drqn import DRQNetwork

    print("=" * 50)
    print("Test 2: DRQNetwork")
    print("=" * 50)

    net = DRQNetwork(in_channels=12, n_actions=19)
    obs = torch.randint(0, 255, (4, 10, 12, 84, 84), dtype=torch.uint8)
    h0 = net.init_hidden(4)

    q, h = net(obs, h0)
    print(f"has_memory: {net.has_memory}")
    print(f"Q shape: {q.shape}")
    print(f"Hidden shape: {h.shape}")
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    assert q.shape == (4, 10, 19)
    assert h.shape == (1, 4, 256)
    print("PASSED\n")


def test_actor_critic():
    from src.networks.actor_critic import RecurrentActorCritic

    print("=" * 50)
    print("Test 3: RecurrentActorCritic")
    print("=" * 50)

    net = RecurrentActorCritic(in_channels=12, n_actions=19)
    obs = torch.randint(0, 255, (8, 12, 84, 84), dtype=torch.uint8)
    h0 = net.init_hidden(8)

    logits, value, h = net(obs, h0)
    print(f"has_memory: {net.has_memory}")
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Hidden shape: {h.shape}")
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    assert logits.shape == (8, 19)
    assert value.shape == (8,)
    assert h.shape == (8, 256)
    print("PASSED\n")


def test_shared_encoder():
    """Verify all networks use the same encoder architecture."""
    from src.networks.dqn import DQNetwork
    from src.networks.drqn import DRQNetwork
    from src.networks.actor_critic import RecurrentActorCritic

    print("=" * 50)
    print("Test 4: Shared encoder architecture")
    print("=" * 50)

    dqn = DQNetwork(in_channels=12, n_actions=19)
    drqn = DRQNetwork(in_channels=12, n_actions=19)
    ac = RecurrentActorCritic(in_channels=12, n_actions=19)

    assert dqn._encoder_out_size == drqn._encoder_out_size == ac._encoder_out_size
    print(f"Encoder output size: {dqn._encoder_out_size}")

    # same input → same encoder output shape
    obs = torch.randint(0, 255, (1, 12, 84, 84), dtype=torch.uint8)
    e1 = dqn.encode(obs)
    e2 = drqn.encode(obs)
    e3 = ac.encode(obs)
    assert e1.shape == e2.shape == e3.shape
    print(f"Encoded shape: {e1.shape}")
    print("PASSED\n")


def test_env_compatibility():
    """Verify networks work with env's in_channels and n_actions."""
    from src.envs.deathmatch import DeathmatchEnv, EnvConfig
    from src.networks.dqn import DQNetwork
    from src.networks.drqn import DRQNetwork
    from src.networks.actor_critic import RecurrentActorCritic

    print("=" * 50)
    print("Test 5: Environment compatibility")
    print("=" * 50)

    env = DeathmatchEnv(EnvConfig(scenario="deathmatch", mode="single"))
    obs = env.reset()

    print(f"Env in_channels: {env.in_channels}, n_actions: {env.n_actions}")

    obs_t = torch.from_numpy(obs).unsqueeze(0)

    # DQN
    dqn = DQNetwork(env.in_channels, env.n_actions)
    q, _ = dqn(obs_t)
    assert q.shape == (1, env.n_actions)

    # DRQN
    drqn = DRQNetwork(env.in_channels, env.n_actions)
    q, _ = drqn(obs_t.unsqueeze(1), drqn.init_hidden(1))
    assert q.shape == (1, 1, env.n_actions)

    # Actor-Critic
    ac = RecurrentActorCritic(env.in_channels, env.n_actions)
    logits, value, h = ac(obs_t, ac.init_hidden(1))
    assert logits.shape == (1, env.n_actions)

    print("All networks compatible with env")
    print("PASSED\n")

    env.close()


if __name__ == "__main__":
    test_dqn()
    test_drqn()
    test_actor_critic()
    test_shared_encoder()
    test_env_compatibility()
    print("=" * 50)
    print("All network tests passed!")
    print("=" * 50)