"""Compare 1-step SARSA and n-step SARSA on CliffWalking-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train import Config as TrainConfig
from train import evaluate, greedy_rollout, moving_average, render_policy, train


@dataclass
class CompareConfig:
    """Configuration for comparing 1-step SARSA and n-step SARSA."""

    episodes: int = 800
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_start: float = 0.1
    epsilon_end: float = 0.1
    epsilon_decay: float = 1.0
    eval_episodes: int = 200
    max_steps_per_episode: int = 500
    n_step: int = 4
    seed: int = 42
    moving_avg_window: int = 50
    run_name: str = "one-step-vs-n-step-sarsa"


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(description="Compare 1-step SARSA and n-step SARSA on CliffWalking-v1.")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes per algorithm.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon-start", type=float, default=0.1, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=1.0, help="Per-episode epsilon decay factor.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Hard cap on steps per episode to avoid endless wandering in CliffWalking.",
    )
    parser.add_argument("--n-step", type=int, default=4, help="Number of steps used by the multi-step baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment and action sampling.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=50,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="one-step-vs-n-step-sarsa",
        help="Name of the comparison output subdirectory.",
    )
    args = parser.parse_args()
    if args.n_step < 2:
        parser.error("--n-step must be at least 2 for the comparison script.")
    return CompareConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        n_step=args.n_step,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
    )


def to_train_config(config: CompareConfig, n_step: int, run_name: str) -> TrainConfig:
    """Convert comparison settings into a training configuration."""

    return TrainConfig(
        episodes=config.episodes,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        eval_episodes=config.eval_episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        n_step=n_step,
        seed=config.seed,
        moving_avg_window=config.moving_avg_window,
        run_name=run_name,
        render_final_policy=False,
    )


def save_comparison_curve(
    one_step_rewards: list[float], n_step_rewards: list[float], config: CompareConfig, output_path: Path
) -> None:
    """Save the smoothed reward curves for both algorithms."""

    one_step_smoothed = moving_average(one_step_rewards, config.moving_avg_window)
    n_step_smoothed = moving_average(n_step_rewards, config.moving_avg_window)

    plt.figure(figsize=(10, 4.5))
    plt.plot(one_step_smoothed, linewidth=2, label="1-step SARSA")
    plt.plot(n_step_smoothed, linewidth=2, label=f"{config.n_step}-step SARSA")
    plt.title("CliffWalking: 1-step SARSA vs n-step SARSA")
    plt.xlabel("Episode")
    plt.ylabel(
        f"Moving average reward (window={min(config.moving_avg_window, len(one_step_rewards), len(n_step_rewards))})"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: CompareConfig,
    one_step_q: np.ndarray,
    one_step_rewards: list[float],
    one_step_metrics: dict[str, float],
    n_step_q: np.ndarray,
    n_step_rewards: list[float],
    n_step_metrics: dict[str, float],
) -> Path:
    """Persist comparison plots and summary metadata."""

    output_dir = Path(__file__).resolve().parent / "outputs" / "comparisons" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_curve(
        one_step_rewards=one_step_rewards,
        n_step_rewards=n_step_rewards,
        config=config,
        output_path=output_dir / "comparison_reward_curve.png",
    )

    one_step_config = to_train_config(config, n_step=1, run_name="one-step")
    n_step_config = to_train_config(config, n_step=config.n_step, run_name=f"{config.n_step}-step")
    summary = {
        "config": asdict(config),
        "one_step_sarsa": {
            "train": {
                "episodes": config.episodes,
                "avg_reward_last_100": float(np.mean(one_step_rewards[-100:])) if one_step_rewards else 0.0,
                "best_episode_reward": float(np.max(one_step_rewards)) if one_step_rewards else 0.0,
            },
            "eval": one_step_metrics,
            "policy": render_policy(one_step_q),
            "greedy_rollout": greedy_rollout(one_step_config, one_step_q),
            "q_table": one_step_q.round(6).tolist(),
        },
        "n_step_sarsa": {
            "train": {
                "episodes": config.episodes,
                "avg_reward_last_100": float(np.mean(n_step_rewards[-100:])) if n_step_rewards else 0.0,
                "best_episode_reward": float(np.max(n_step_rewards)) if n_step_rewards else 0.0,
            },
            "eval": n_step_metrics,
            "policy": render_policy(n_step_q),
            "greedy_rollout": greedy_rollout(n_step_config, n_step_q),
            "q_table": n_step_q.round(6).tolist(),
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def print_policy(label: str, policy: list[list[str]]) -> None:
    """Print a labeled greedy policy as a compact 4x12 grid."""

    print(f"{label} greedy policy:")
    for row in policy:
        print(" ".join(row))
    print()


def print_greedy_rollout(label: str, rollout: dict[str, object]) -> None:
    """Print the realized greedy path from the start state."""

    max_items_to_print = 20
    print(
        f"{label} greedy rollout: "
        f"steps={rollout['steps']}, total_reward={rollout['total_reward']:.1f}, cliff_falls={rollout['cliff_falls']}"
    )
    print("Visited states:")
    visited_states = rollout["visited_states"]
    if len(visited_states) <= max_items_to_print:
        print(" -> ".join(visited_states))
    else:
        head = " -> ".join(visited_states[:max_items_to_print])
        print(f"{head} -> ... ({len(visited_states) - max_items_to_print} more states)")
    print("Transitions:")
    transitions = rollout["transitions"]
    for transition in transitions[:max_items_to_print]:
        print(f"  {transition}")
    if len(transitions) > max_items_to_print:
        print(f"  ... ({len(transitions) - max_items_to_print} more transitions)")
    if rollout["truncated"]:
        print("  Stopped early because the max step limit was reached.")
    print()


def main() -> None:
    """Run both algorithms, evaluate them, and save comparison artifacts."""

    config = parse_args()
    one_step_config = to_train_config(config, n_step=1, run_name="one-step-sarsa")
    n_step_config = to_train_config(config, n_step=config.n_step, run_name=f"{config.n_step}-step-sarsa")

    one_step_q, one_step_rewards = train(one_step_config)
    n_step_q, n_step_rewards = train(n_step_config)
    one_step_metrics = evaluate(one_step_config, one_step_q)
    n_step_metrics = evaluate(n_step_config, n_step_q)
    output_dir = save_outputs(
        config=config,
        one_step_q=one_step_q,
        one_step_rewards=one_step_rewards,
        one_step_metrics=one_step_metrics,
        n_step_q=n_step_q,
        n_step_rewards=n_step_rewards,
        n_step_metrics=n_step_metrics,
    )

    print(f"Comparison saved to: {output_dir}")
    print(
        "1-step SARSA eval: "
        f"avg_reward={one_step_metrics['avg_reward']:.4f}, "
        f"avg_steps_to_goal={one_step_metrics['avg_steps_to_goal']:.4f}, "
        f"avg_cliff_falls={one_step_metrics['avg_cliff_falls']:.4f}"
    )
    print(
        f"{config.n_step}-step SARSA eval: "
        f"avg_reward={n_step_metrics['avg_reward']:.4f}, "
        f"avg_steps_to_goal={n_step_metrics['avg_steps_to_goal']:.4f}, "
        f"avg_cliff_falls={n_step_metrics['avg_cliff_falls']:.4f}"
    )
    print()

    print_policy("1-step SARSA", render_policy(one_step_q))
    print_policy(f"{config.n_step}-step SARSA", render_policy(n_step_q))
    print_greedy_rollout("1-step SARSA", greedy_rollout(one_step_config, one_step_q))
    print_greedy_rollout(f"{config.n_step}-step SARSA", greedy_rollout(n_step_config, n_step_q))


if __name__ == "__main__":
    main()
