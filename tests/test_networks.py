"""
Test script for networks.
Run: PYTHONPATH=. python tests/test_networks.py
"""

import sys
from pathlib import Path
import numpy as np

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

def test_agent_ppo():
    """Test Agent in PPO/IPPO mode (local critic)."""
    from src.networks.actor_critic import RecurrentActorCritic
    from src.networks.agent import Agent

    print("=" * 50)
    print("Test: Agent (PPO/IPPO mode)")
    print("=" * 50)

    ac = RecurrentActorCritic(in_channels=12, n_actions=19)
    agent = Agent(ac)

    assert not agent.is_mappo
    print(f"  is_mappo: {agent.is_mappo}")
    print(f"  params: {sum(p.numel() for p in agent.parameters()):,}")

    # act
    obs = np.random.randint(0, 255, (12, 84, 84), dtype=np.uint8)
    h = agent.init_hidden(1)
    action, logprob, value, h, ch = agent.act(obs, h)
    print(f"  act: action={action}, logprob={logprob:.3f}, value={value:.3f}")

    assert isinstance(action, int)
    assert 0 <= action < 19

    # get_value
    v = agent.get_value(obs, h)
    print(f"  get_value: {v:.3f}")

    # evaluate_actions
    agent.train()
    obs_seq = torch.randint(0, 255, (4, 10, 12, 84, 84), dtype=torch.uint8)
    actions_seq = torch.randint(0, 19, (4, 10))
    h_init = agent.init_hidden(4)
    dones = torch.zeros(4, 10)

    logprobs, values, entropy = agent.evaluate_actions(
        obs_seq, actions_seq, h_init, dones)

    print(f"  evaluate: logprobs={logprobs.shape}, values={values.shape}, entropy={entropy:.3f}")
    assert logprobs.shape == (4, 10)
    assert values.shape == (4, 10)

    # save/load
    agent.save("/tmp/test_agent_ppo.pt")
    agent2 = Agent(RecurrentActorCritic(in_channels=12, n_actions=19))
    agent2.load("/tmp/test_agent_ppo.pt")
    print("  save/load: OK")

    print("PASSED\n")


def test_agent_mappo():
    """Test Agent in MAPPO mode (centralized critic)."""
    from src.networks.actor_critic import RecurrentActorCritic
    from src.networks.centralized_critic import CentralizedCritic
    from src.networks.agent import Agent

    print("=" * 50)
    print("Test: Agent (MAPPO mode)")
    print("=" * 50)

    n_agents = 4
    in_channels = 12

    ac = RecurrentActorCritic(in_channels=in_channels, n_actions=19)
    cc = CentralizedCritic(n_agents=n_agents, in_channels=in_channels)
    agent = Agent(ac, centralized_critic=cc)

    assert agent.is_mappo
    print(f"  is_mappo: {agent.is_mappo}")
    print(f"  actor params: {sum(p.numel() for p in ac.parameters()):,}")
    print(f"  critic params: {sum(p.numel() for p in cc.parameters()):,}")
    print(f"  total params: {sum(p.numel() for p in agent.parameters()):,}")

    # act with global obs
    obs = np.random.randint(0, 255, (in_channels, 84, 84), dtype=np.uint8)
    global_obs = np.random.randint(0, 255, (n_agents * in_channels, 84, 84), dtype=np.uint8)
    ah = agent.init_hidden(1)
    ch = agent.init_critic_hidden(1)

    action, logprob, value, ah, ch = agent.act(obs, ah, ch, global_obs)
    print(f"  act: action={action}, logprob={logprob:.3f}, value={value:.3f}")

    # get_value centralized
    v = agent.get_value(obs, ah, ch, global_obs)
    print(f"  get_value (centralized): {v:.3f}")

    # evaluate_actions with global obs
    agent.train()
    B, T = 4, 10
    obs_seq = torch.randint(0, 255, (B, T, in_channels, 84, 84), dtype=torch.uint8)
    global_seq = torch.randint(0, 255, (B, T, n_agents * in_channels, 84, 84), dtype=torch.uint8)
    actions_seq = torch.randint(0, 19, (B, T))
    ah_init = agent.init_hidden(B)
    ch_init = agent.init_critic_hidden(B)
    dones = torch.zeros(B, T)

    logprobs, values, entropy = agent.evaluate_actions(
        obs_seq, actions_seq, ah_init, dones, ch_init, global_seq)

    print(f"  evaluate: logprobs={logprobs.shape}, values={values.shape}, entropy={entropy:.3f}")
    assert logprobs.shape == (B, T)
    assert values.shape == (B, T)

    # save/load
    agent.save("/tmp/test_agent_mappo.pt")
    agent2 = Agent(
        RecurrentActorCritic(in_channels=in_channels, n_actions=19),
        CentralizedCritic(n_agents=n_agents, in_channels=in_channels),
    )
    agent2.load("/tmp/test_agent_mappo.pt")
    print("  save/load: OK")

    # transfer: load only actor from PPO pretrained
    ppo_agent = Agent(RecurrentActorCritic(in_channels=in_channels, n_actions=19))
    ppo_agent.save("/tmp/test_ppo_pretrained.pt")

    mappo_agent = Agent(
        RecurrentActorCritic(in_channels=in_channels, n_actions=19),
        CentralizedCritic(n_agents=n_agents, in_channels=in_channels),
    )
    mappo_agent.load_pretrained_actor("/tmp/test_ppo_pretrained.pt")
    print("  transfer (PPO → MAPPO actor): OK")

    print("PASSED\n")


if __name__ == "__main__":
    test_dqn()
    test_drqn()
    test_actor_critic()
    test_shared_encoder()
    test_env_compatibility()
    test_agent_mappo()
    test_agent_ppo()
    print("=" * 50)
    print("All network tests passed!")
    print("=" * 50)