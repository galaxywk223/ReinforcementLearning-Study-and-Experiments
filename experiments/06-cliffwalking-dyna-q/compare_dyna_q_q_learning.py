"""Compare tabular Dyna-Q against plain Q-learning on CliffWalking-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train import Config, evaluate, moving_average, run_training


@dataclass
class CompareConfig:
    """Configuration for a Dyna-Q versus Q-learning comparison run."""

    episodes: int = 400
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_start: float = 0.1
    epsilon_end: float = 0.1
    epsilon_decay: float = 1.0
    planning_steps: int = 10
    eval_episodes: int = 200
    max_steps_per_episode: int = 500
    seed: int = 42
    moving_avg_window: int = 40
    run_name: str = "dyna-q-vs-q-learning"


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(
        description="Train Dyna-Q and plain Q-learning on CliffWalking-v1, then save the comparison artifacts."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes per algorithm.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon-start", type=float, default=0.1, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=1.0, help="Per-episode epsilon decay factor.")
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=10,
        help="Number of model rollouts used by the Dyna-Q baseline after each real step.",
    )
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes per algorithm.")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Hard cap on steps per episode to avoid endless wandering in CliffWalking.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared by both algorithms.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=40,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="dyna-q-vs-q-learning",
        help="Name of the comparison output subdirectory.",
    )
    args = parser.parse_args()
    return CompareConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        planning_steps=args.planning_steps,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
    )


def to_train_config(config: CompareConfig, run_name: str) -> Config:
    """Convert comparison settings into a training configuration."""

    return Config(
        episodes=config.episodes,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        planning_steps=config.planning_steps,
        eval_episodes=config.eval_episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        seed=config.seed,
        moving_avg_window=config.moving_avg_window,
        run_name=run_name,
        render_final_policy=False,
    )


def save_comparison_curve(
    q_learning_rewards: list[float],
    dyna_q_rewards: list[float],
    window: int,
    output_path: Path,
) -> None:
    """Save a smoothed comparison curve for both algorithms."""

    q_smoothed = moving_average(q_learning_rewards, window)
    dyna_smoothed = moving_average(dyna_q_rewards, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(q_smoothed, linewidth=2, label="Q-Learning")
    plt.plot(dyna_smoothed, linewidth=2, label="Dyna-Q")
    plt.title("CliffWalking Q-Learning vs Dyna-Q")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(q_learning_rewards), len(dyna_q_rewards))})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: CompareConfig,
    q_learning_rewards: list[float],
    dyna_q_rewards: list[float],
    q_learning_train: dict[str, float],
    dyna_q_train: dict[str, float],
    q_learning_eval: dict[str, float],
    dyna_q_eval: dict[str, float],
) -> Path:
    """Persist comparison plots and summary metadata."""

    output_dir = Path(__file__).resolve().parent / "outputs" / "comparisons" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_curve(
        q_learning_rewards=q_learning_rewards,
        dyna_q_rewards=dyna_q_rewards,
        window=config.moving_avg_window,
        output_path=output_dir / "comparison_reward_curve.png",
    )
    summary = {
        "config": asdict(config),
        "q_learning": {
            "train": q_learning_train,
            "eval": q_learning_eval,
        },
        "dyna_q": {
            "train": dyna_q_train,
            "eval": dyna_q_eval,
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
    q_learning_config = to_train_config(config, run_name="q-learning-reference")
    dyna_q_config = to_train_config(config, run_name="dyna-q-reference")

    q_learning_table, q_learning_rewards, _, q_learning_train = run_training(q_learning_config, planning_steps=0)
    dyna_q_table, dyna_q_rewards, _, dyna_q_train = run_training(
        dyna_q_config,
        planning_steps=config.planning_steps,
    )
    q_learning_eval = evaluate(q_learning_config, q_learning_table)
    dyna_q_eval = evaluate(dyna_q_config, dyna_q_table)
    output_dir = save_outputs(
        config=config,
        q_learning_rewards=q_learning_rewards,
        dyna_q_rewards=dyna_q_rewards,
        q_learning_train=q_learning_train,
        dyna_q_train=dyna_q_train,
        q_learning_eval=q_learning_eval,
        dyna_q_eval=dyna_q_eval,
    )

    print(f"Comparison saved to: {output_dir}")
    print(f"Q-Learning avg_reward: {q_learning_eval['avg_reward']:.4f}")
    print(f"Dyna-Q avg_reward: {dyna_q_eval['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
