"""Train a synchronous actor-critic agent on CartPole-v1."""

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
    """Configuration for a single CartPole actor-critic run."""

    episodes: int = 400
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    eval_episodes: int = 50
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "cartpole-actor-critic-baseline"
    print_eval_rollout: bool = False


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a synchronous actor-critic agent on CartPole-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by both MLP layers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for TD targets.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="Weight applied to the critic regression term.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient used to keep early exploration alive.",
    )
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
        default="cartpole-actor-critic-baseline",
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
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
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


def train(config: Config) -> tuple[ActorCriticNetwork, list[float], list[float], dict[str, float]]:
    """Train the actor-critic network and return reward and critic-loss histories."""

    set_global_seeds(config.seed)
    device = select_device()
    env = make_env(config.seed)
    model = ActorCriticNetwork(4, 2, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    episode_rewards: list[float] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    entropies: list[float] = []

    for episode_index in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        episode_actor_losses: list[float] = []
        episode_critic_losses: list[float] = []
        episode_entropies: list[float] = []

        while not (terminated or truncated):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(state_tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))

            with torch.no_grad():
                next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                _, next_value = model(next_state_tensor)
                td_target = torch.tensor(
                    [reward],
                    dtype=torch.float32,
                    device=device,
                ) + config.gamma * next_value * float(not (terminated or truncated))

            advantage = td_target - value
            actor_loss = -(distribution.log_prob(action) * advantage.detach()).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            entropy_bonus = distribution.entropy().mean()
            loss = actor_loss + config.value_loss_coef * critic_loss - config.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            state = next_state
            episode_actor_losses.append(float(actor_loss.item()))
            episode_critic_losses.append(float(critic_loss.item()))
            episode_entropies.append(float(entropy_bonus.item()))

        episode_rewards.append(total_reward)
        actor_losses.append(float(np.mean(episode_actor_losses)))
        critic_losses.append(float(np.mean(episode_critic_losses)))
        entropies.append(float(np.mean(episode_entropies)))

    env.close()
    train_stats = {
        "avg_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        "best_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "avg_actor_loss_last_20": float(np.mean(actor_losses[-20:])) if actor_losses else 0.0,
        "avg_critic_loss_last_20": float(np.mean(critic_losses[-20:])) if critic_losses else 0.0,
        "avg_entropy_last_20": float(np.mean(entropies[-20:])) if entropies else 0.0,
        "device": str(device),
    }
    return model, episode_rewards, critic_losses, train_stats


def evaluate(config: Config, model: ActorCriticNetwork) -> tuple[dict[str, float], dict[str, object]]:
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
                logits, _ = model(state_tensor)
                action = int(torch.argmax(logits.squeeze(0)).item())
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
    model: ActorCriticNetwork,
    episode_rewards: list[float],
    critic_losses: list[float],
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
        title="CartPole Actor-Critic Reward Curve",
        ylabel=f"Moving average reward (window={min(config.moving_avg_window, len(episode_rewards))})",
        output_path=output_dir / "reward_curve.png",
    )
    save_curve(
        values=critic_losses,
        window=config.moving_avg_window,
        title="CartPole Actor-Critic Critic Loss Curve",
        ylabel=f"Moving average critic loss (window={min(config.moving_avg_window, len(critic_losses))})",
        output_path=output_dir / "critic_loss_curve.png",
    )

    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": eval_metrics,
        "sample_eval_rollout": sample_rollout,
        "model": {
            "architecture": f"SharedMLP(4 -> {config.hidden_dim} -> {config.hidden_dim}) + policy/value heads",
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
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
    """Run actor-critic training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    model, episode_rewards, critic_losses, train_stats = train(config)
    eval_metrics, sample_rollout = evaluate(config, model)
    output_dir = save_outputs(
        config=config,
        model=model,
        episode_rewards=episode_rewards,
        critic_losses=critic_losses,
        train_stats=train_stats,
        eval_metrics=eval_metrics,
        sample_rollout=sample_rollout,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_20: {train_stats['avg_reward_last_20']:.4f}")
    print(f"Training avg_critic_loss_last_20: {train_stats['avg_critic_loss_last_20']:.6f}")
    print(f"Evaluation avg_reward: {eval_metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_episode_length: {eval_metrics['avg_episode_length']:.4f}")
    print(f"Evaluation success_rate: {eval_metrics['success_rate']:.4f}")

    if config.print_eval_rollout:
        print_rollout(sample_rollout)


if __name__ == "__main__":
    main()
