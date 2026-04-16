"""Trace a single DQN minibatch update on scripted CartPole transitions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from train import QNetwork


@dataclass
class TraceConfig:
    """Configuration for the scripted DQN trace."""

    gamma: float = 0.99
    hidden_dim: int = 128
    lr: float = 1e-3
    seed: int = 42


SCRIPTED_BATCH = [
    {
        "state": [0.020000, 0.100000, 0.030000, 0.150000],
        "action": 1,
        "reward": 1.0,
        "next_state": [0.022000, 0.290000, 0.033000, -0.120000],
        "done": False,
    },
    {
        "state": [0.010000, -0.180000, 0.020000, -0.250000],
        "action": 0,
        "reward": 1.0,
        "next_state": [0.006000, -0.370000, 0.015000, 0.040000],
        "done": False,
    },
    {
        "state": [0.045000, 0.210000, 0.200000, 0.500000],
        "action": 1,
        "reward": 1.0,
        "next_state": [0.049000, 0.410000, 0.210000, 0.820000],
        "done": True,
    },
    {
        "state": [-0.030000, 0.050000, -0.040000, -0.100000],
        "action": 0,
        "reward": 1.0,
        "next_state": [-0.029000, -0.140000, -0.042000, 0.180000],
        "done": False,
    },
]


def parse_args() -> TraceConfig:
    parser = argparse.ArgumentParser(description="Trace one scripted DQN minibatch update for CartPole.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in the TD target.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by the Q-network.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used for the update step.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to initialize the networks.")
    args = parser.parse_args()
    return TraceConfig(gamma=args.gamma, hidden_dim=args.hidden_dim, lr=args.lr, seed=args.seed)


def format_vector(values: np.ndarray) -> str:
    """Format a vector compactly for CLI output."""

    rounded = np.round(values.astype(np.float64), 6)
    return np.array2string(rounded, precision=6, separator=", ")


def main() -> None:
    """Run the CLI trace entrypoint."""

    config = parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    policy_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=config.hidden_dim)
    target_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=config.hidden_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.lr)

    states = torch.as_tensor([item["state"] for item in SCRIPTED_BATCH], dtype=torch.float32)
    actions = torch.as_tensor([item["action"] for item in SCRIPTED_BATCH], dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor([item["reward"] for item in SCRIPTED_BATCH], dtype=torch.float32)
    next_states = torch.as_tensor([item["next_state"] for item in SCRIPTED_BATCH], dtype=torch.float32)
    dones = torch.as_tensor([item["done"] for item in SCRIPTED_BATCH], dtype=torch.float32)

    predicted_all_before = policy_net(states)
    predicted_selected_before = predicted_all_before.gather(1, actions).squeeze(1)
    next_q = target_net(next_states).detach().max(dim=1).values
    td_target = rewards + config.gamma * next_q * (1.0 - dones)
    loss_before = nn.functional.smooth_l1_loss(predicted_selected_before, td_target)

    print("CartPole DQN scripted minibatch trace")
    print("This script shows one replay minibatch, its TD targets, and one optimizer step.")
    print()

    for index, item in enumerate(SCRIPTED_BATCH, start=1):
        print(f"Sample {index}")
        print(f"  state      = {item['state']}")
        print(f"  action     = {item['action']}")
        print(f"  reward     = {item['reward']:.1f}")
        print(f"  next_state = {item['next_state']}")
        print(f"  done       = {item['done']}")
        print(f"  Q_online(s, .)     = {format_vector(predicted_all_before[index - 1].detach().numpy())}")
        print(f"  Q_pred(selected)   = {predicted_selected_before[index - 1].item():.6f}")
        print(f"  max_next_Q_target  = {next_q[index - 1].item():.6f}")
        print(f"  TD target          = {td_target[index - 1].item():.6f}")
        print()

    print(f"Loss before update: {loss_before.item():.6f}")

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    predicted_all_after = policy_net(states).detach()
    predicted_selected_after = predicted_all_after.gather(1, actions).squeeze(1)
    loss_after = nn.functional.smooth_l1_loss(predicted_selected_after, td_target)

    print(f"Loss after update:  {loss_after.item():.6f}")
    print()
    print("Selected-action Q-values after one optimizer step:")
    for index in range(len(SCRIPTED_BATCH)):
        print(
            f"  Sample {index + 1}: "
            f"before={predicted_selected_before[index].item():.6f}, "
            f"after={predicted_selected_after[index].item():.6f}, "
            f"target={td_target[index].item():.6f}"
        )


if __name__ == "__main__":
    main()
