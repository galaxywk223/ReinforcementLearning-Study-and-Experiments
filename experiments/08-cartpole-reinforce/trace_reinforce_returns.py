"""Print reward-to-go targets and loss weights for a toy REINFORCE episode."""

from __future__ import annotations

import argparse
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace reward-to-go computation for a toy REINFORCE episode."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in reward-to-go targets.")
    return parser.parse_args()


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute standardized reward-to-go returns for one episode."""

    returns = torch.zeros(len(rewards), dtype=torch.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)


def main() -> None:
    """Print one toy episode's returns and weighted log-prob terms."""

    args = parse_args()
    rewards = [1.0, 1.0, 1.0, 1.0]
    log_probs = torch.tensor([-0.35, -0.72, -0.42, -1.10], dtype=torch.float32)
    returns = discounted_returns(rewards, args.gamma)

    print(f"{'t':>2} {'reward':>8} {'return':>10} {'log_prob':>10} {'-log_prob*G':>14}")
    for index, (reward, value, log_prob) in enumerate(zip(rewards, returns, log_probs), start=1):
        contribution = -log_prob.item() * value.item()
        print(f"{index:>2} {reward:>8.2f} {value.item():>10.4f} {log_prob.item():>10.4f} {contribution:>14.4f}")


if __name__ == "__main__":
    main()
