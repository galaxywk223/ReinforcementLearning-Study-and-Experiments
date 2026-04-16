"""Compare actor-critic against REINFORCE on CartPole-v1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from train import Config as ActorCriticConfig
from train import evaluate as evaluate_actor_critic
from train import moving_average, train as train_actor_critic


@dataclass
class CompareConfig:
    """Configuration for an actor-critic versus REINFORCE comparison run."""

    episodes: int = 400
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    eval_episodes: int = 50
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "actor-critic-vs-reinforce"


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(
        description="Train actor-critic and REINFORCE on CartPole-v1, then save the comparison artifacts."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes per algorithm.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by both MLP layers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns and TD targets.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate shared by both algorithms.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation episodes per algorithm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared by both algorithms.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=20,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="actor-critic-vs-reinforce",
        help="Name of the comparison output subdirectory.",
    )
    args = parser.parse_args()
    return CompareConfig(
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
    )


def load_reinforce_module():
    """Load the sibling REINFORCE training module without turning the repo into a package."""

    module_path = Path(__file__).resolve().parents[1] / "08-cartpole-reinforce" / "train.py"
    spec = importlib.util.spec_from_file_location("cartpole_reinforce_train", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load REINFORCE module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_comparison_curve(
    reinforce_rewards: list[float],
    actor_critic_rewards: list[float],
    window: int,
    output_path: Path,
) -> None:
    """Save a smoothed comparison curve for both algorithms."""

    reinforce_smoothed = moving_average(reinforce_rewards, window)
    actor_critic_smoothed = moving_average(actor_critic_rewards, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(reinforce_smoothed, linewidth=2, label="REINFORCE")
    plt.plot(actor_critic_smoothed, linewidth=2, label="Actor-Critic")
    plt.title("CartPole REINFORCE vs Actor-Critic")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(reinforce_rewards), len(actor_critic_rewards))})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: CompareConfig,
    reinforce_train: dict[str, float],
    actor_critic_train: dict[str, float],
    reinforce_eval: dict[str, float],
    actor_critic_eval: dict[str, float],
    reinforce_rewards: list[float],
    actor_critic_rewards: list[float],
) -> Path:
    """Persist comparison plots and summary metadata."""

    output_dir = Path(__file__).resolve().parent / "outputs" / "comparisons" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_curve(
        reinforce_rewards=reinforce_rewards,
        actor_critic_rewards=actor_critic_rewards,
        window=config.moving_avg_window,
        output_path=output_dir / "comparison_reward_curve.png",
    )
    summary = {
        "config": asdict(config),
        "reinforce": {
            "train": reinforce_train,
            "eval": reinforce_eval,
        },
        "actor_critic": {
            "train": actor_critic_train,
            "eval": actor_critic_eval,
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    """Run both algorithms, evaluate them, and save comparison artifacts."""

    config = parse_args()
    reinforce_module = load_reinforce_module()
    reinforce_config = reinforce_module.Config(
        episodes=config.episodes,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        lr=config.lr,
        eval_episodes=config.eval_episodes,
        seed=config.seed,
        moving_avg_window=config.moving_avg_window,
        run_name="reinforce-reference",
        print_eval_rollout=False,
    )
    actor_critic_config = ActorCriticConfig(
        episodes=config.episodes,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        lr=config.lr,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        eval_episodes=config.eval_episodes,
        seed=config.seed,
        moving_avg_window=config.moving_avg_window,
        run_name="actor-critic-reference",
        print_eval_rollout=False,
    )

    reinforce_model, reinforce_rewards, _, reinforce_train_stats = reinforce_module.train(reinforce_config)
    actor_critic_model, actor_critic_rewards, _, actor_critic_train_stats = train_actor_critic(actor_critic_config)
    reinforce_eval, _ = reinforce_module.evaluate(reinforce_config, reinforce_model)
    actor_critic_eval, _ = evaluate_actor_critic(actor_critic_config, actor_critic_model)
    output_dir = save_outputs(
        config=config,
        reinforce_train=reinforce_train_stats,
        actor_critic_train=actor_critic_train_stats,
        reinforce_eval=reinforce_eval,
        actor_critic_eval=actor_critic_eval,
        reinforce_rewards=reinforce_rewards,
        actor_critic_rewards=actor_critic_rewards,
    )

    print(f"Comparison saved to: {output_dir}")
    print(f"REINFORCE avg_reward: {reinforce_eval['avg_reward']:.4f}")
    print(f"Actor-Critic avg_reward: {actor_critic_eval['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
