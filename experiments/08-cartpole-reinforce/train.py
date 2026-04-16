"""Train a REINFORCE agent on CartPole-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class Config:
    """Configuration for a single CartPole REINFORCE run."""

    episodes: int = 400
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    eval_episodes: int = 50
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "cartpole-reinforce-baseline"
    print_eval_rollout: bool = False


class PolicyNetwork(nn.Module):
    """Two-layer MLP policy used by REINFORCE."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE agent on CartPole-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by both MLP layers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for reward-to-go targets.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training and evaluation.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=20,
        help="Window size used when smoothing training curves.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="cartpole-reinforce-baseline",
        help="Name of the output subdirectory.",
    )
    parser.add_argument(
        "--print-eval-rollout",
        action="store_true",
        help="Print a sample greedy evaluation rollout after training.",
    )
    args = parser.parse_args()
    return Config(
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        print_eval_rollout=args.print_eval_rollout,
    )


def make_env(seed: int) -> gym.Env:
    """Create a seeded CartPole environment."""

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def set_global_seeds(seed: int) -> None:
    """Set random seeds for NumPy and PyTorch."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Prefer CUDA when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute reward-to-go returns for one complete episode."""

    returns = torch.zeros(len(rewards), dtype=torch.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    # Standardization reduces variance without introducing a value baseline.
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def train(config: Config) -> tuple[PolicyNetwork, list[float], list[float], dict[str, float]]:
    """Train the policy network and return reward and loss histories."""

    set_global_seeds(config.seed)
    device = select_device()
    env = make_env(config.seed)
    policy_net = PolicyNetwork(4, 2, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.lr)
    episode_rewards: list[float] = []
    losses: list[float] = []
    entropies: list[float] = []

    for episode_index in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode_index)
        terminated = False
        truncated = False
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        episode_entropies: list[torch.Tensor] = []
        total_reward = 0.0

        while not (terminated or truncated):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy_net(state_tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))

            rewards.append(float(reward))
            log_probs.append(distribution.log_prob(action).squeeze(0))
            episode_entropies.append(distribution.entropy().squeeze(0))
            total_reward += reward
            state = next_state

        returns = discounted_returns(rewards, config.gamma).to(device)
        policy_loss = -(torch.stack(log_probs) * returns).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)
        losses.append(float(policy_loss.item()))
        entropies.append(float(torch.stack(episode_entropies).mean().item()))

    env.close()
    train_stats = {
        "avg_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        "best_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "avg_loss_last_20": float(np.mean(losses[-20:])) if losses else 0.0,
        "avg_entropy_last_20": float(np.mean(entropies[-20:])) if entropies else 0.0,
        "device": str(device),
    }
    return policy_net, episode_rewards, losses, train_stats


def evaluate(config: Config, policy_net: PolicyNetwork) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate the greedy policy and record one representative rollout."""

    device = select_device()
    env = make_env(config.seed + 1)
    max_episode_steps = int(env.spec.max_episode_steps) if env.spec is not None else 500
    rewards: list[float] = []
    episode_lengths: list[int] = []
    successes = 0
    sample_rollout: dict[str, object] | None = None

    for episode_index in range(config.eval_episodes):
        state, _ = env.reset(seed=config.seed + 1 + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        transitions: list[dict[str, object]] = []

        while not (terminated or truncated):
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy_net(state_tensor).squeeze(0)
                action = int(torch.argmax(logits).item())
            next_state, reward, terminated, truncated, _ = env.step(action)

            if sample_rollout is None and len(transitions) < 25:
                transitions.append(
                    {
                        "step": steps + 1,
                        "state": np.asarray(state, dtype=np.float32).round(6).tolist(),
                        "action": action,
                        "reward": float(reward),
                        "next_state": np.asarray(next_state, dtype=np.float32).round(6).tolist(),
                    }
                )

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(float(total_reward))
        episode_lengths.append(steps)
        if steps >= max_episode_steps:
            successes += 1

        if sample_rollout is None:
            sample_rollout = {
                "steps": steps,
                "total_reward": float(total_reward),
                "truncated": bool(truncated),
                "terminated": bool(terminated),
                "transitions": transitions,
            }

    env.close()
    assert sample_rollout is not None
    metrics = {
        "avg_reward": float(np.mean(rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(successes / config.eval_episodes),
    }
    return metrics, sample_rollout


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Compute a simple moving average over a history list."""

    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return data
    window = max(1, min(window, len(data)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode="valid")


def save_curve(values: list[float], window: int, title: str, ylabel: str, output_path: Path) -> None:
    """Save a smoothed curve image."""

    smoothed = moving_average(values, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(smoothed, linewidth=2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: Config,
    policy_net: PolicyNetwork,
    episode_rewards: list[float],
    losses: list[float],
    train_stats: dict[str, float],
    eval_metrics: dict[str, float],
    sample_rollout: dict[str, object],
) -> Path:
    """Persist plots, model metadata, and summary JSON."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_curve(
        values=episode_rewards,
        window=config.moving_avg_window,
        title="CartPole REINFORCE Reward Curve",
        ylabel=f"Moving average reward (window={min(config.moving_avg_window, len(episode_rewards))})",
        output_path=output_dir / "reward_curve.png",
    )
    save_curve(
        values=losses,
        window=config.moving_avg_window,
        title="CartPole REINFORCE Loss Curve",
        ylabel=f"Moving average policy loss (window={min(config.moving_avg_window, len(losses))})",
        output_path=output_dir / "loss_curve.png",
    )

    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": eval_metrics,
        "sample_eval_rollout": sample_rollout,
        "model": {
            "architecture": f"MLP(4 -> {config.hidden_dim} -> {config.hidden_dim} -> 2)",
            "parameter_count": int(sum(parameter.numel() for parameter in policy_net.parameters())),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_rollout(rollout: dict[str, object]) -> None:
    """Print a compact version of a sampled evaluation rollout."""

    print(
        "Sample greedy rollout: "
        f"steps={rollout['steps']}, total_reward={rollout['total_reward']:.1f}, "
        f"terminated={rollout['terminated']}, truncated={rollout['truncated']}"
    )
    for transition in rollout["transitions"]:
        print(
            f"  Step {transition['step']}: "
            f"state={transition['state']} -> action={transition['action']} -> "
            f"reward={transition['reward']:.1f} -> next_state={transition['next_state']}"
        )


def main() -> None:
    """Run REINFORCE training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    policy_net, episode_rewards, losses, train_stats = train(config)
    eval_metrics, sample_rollout = evaluate(config, policy_net)
    output_dir = save_outputs(
        config=config,
        policy_net=policy_net,
        episode_rewards=episode_rewards,
        losses=losses,
        train_stats=train_stats,
        eval_metrics=eval_metrics,
        sample_rollout=sample_rollout,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_20: {train_stats['avg_reward_last_20']:.4f}")
    print(f"Training avg_loss_last_20: {train_stats['avg_loss_last_20']:.6f}")
    print(f"Evaluation avg_reward: {eval_metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_episode_length: {eval_metrics['avg_episode_length']:.4f}")
    print(f"Evaluation success_rate: {eval_metrics['success_rate']:.4f}")

    if config.print_eval_rollout:
        print_rollout(sample_rollout)


if __name__ == "__main__":
    main()
