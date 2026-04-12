"""Train a tabular Q-learning agent on FrozenLake-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    """Configuration for a single FrozenLake training run."""

    episodes: int = 4000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999
    eval_episodes: int = 200
    seed: int = 42
    is_slippery: bool = True
    moving_avg_window: int = 100
    run_name: str = "q-learning-baseline"
    render_final_policy: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on FrozenLake-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=4000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=0.999, help="Per-episode epsilon decay factor.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment and action sampling.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=100,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument("--run-name", type=str, default="q-learning-baseline", help="Name of the output subdirectory.")
    parser.add_argument(
        "--non-slippery",
        action="store_true",
        help="Use the deterministic FrozenLake variant instead of the slippery default.",
    )
    parser.add_argument(
        "--render-final-policy",
        action="store_true",
        help="Print the final greedy policy after training.",
    )
    args = parser.parse_args()
    return Config(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        is_slippery=not args.non_slippery,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        render_final_policy=args.render_final_policy,
    )


def make_env(is_slippery: bool, seed: int) -> gym.Env:
    """Create a seeded FrozenLake environment."""

    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def epsilon_greedy_action(
    q_table: np.ndarray, state: int, epsilon: float, action_space: gym.Space, rng: np.random.Generator
) -> int:
    """Choose an action using epsilon-greedy exploration."""

    if rng.random() < epsilon:
        return int(action_space.sample())
    return int(np.argmax(q_table[state]))


def train(config: Config) -> tuple[np.ndarray, list[float]]:
    """Train the agent and return the learned Q-table and raw episode rewards."""

    env = make_env(is_slippery=config.is_slippery, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    # q_table[s, a] stores the current estimate of the long-term value of
    # taking action a from state s.
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    episode_rewards: list[float] = []
    epsilon = config.epsilon_start

    for _ in range(config.episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            td_target = reward + config.gamma * np.max(q_table[next_state]) * (not terminated)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += config.alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    env.close()
    return q_table, episode_rewards


def evaluate(config: Config, q_table: np.ndarray) -> dict[str, float]:
    """Evaluate the greedy policy induced by the learned Q-table."""

    env = make_env(is_slippery=config.is_slippery, seed=config.seed + 1)
    rewards: list[float] = []

    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(np.asarray(rewards) > 0.0)),
    }


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Compute a simple moving average over the reward history."""

    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return data
    window = max(1, min(window, len(data)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode="valid")


def save_reward_curve(rewards: list[float], window: int, output_path: Path) -> None:
    """Save the smoothed reward curve for a training run."""

    smoothed = moving_average(rewards, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(smoothed, linewidth=2)
    plt.title("FrozenLake Q-Learning Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(rewards))})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def render_policy(q_table: np.ndarray) -> list[list[str]]:
    """Convert the greedy policy into a 4x4 grid of direction symbols."""

    action_symbols = ["L", "D", "R", "U"]
    best_actions = np.argmax(q_table, axis=1)
    symbols = [action_symbols[action] for action in best_actions]
    return [symbols[index : index + 4] for index in range(0, len(symbols), 4)]


def save_outputs(config: Config, q_table: np.ndarray, rewards: list[float], metrics: dict[str, float]) -> Path:
    """Persist plots and summary metadata for a training run."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_reward_curve(rewards, config.moving_avg_window, output_dir / "reward_curve.png")
    summary = {
        "config": asdict(config),
        "train": {
            "episodes": config.episodes,
            "avg_reward_last_100": float(np.mean(rewards[-100:])) if rewards else 0.0,
            "best_episode_reward": float(np.max(rewards)) if rewards else 0.0,
        },
        "eval": metrics,
        "policy": render_policy(q_table),
        "q_table": q_table.round(6).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_policy(policy: list[list[str]]) -> None:
    """Print the greedy policy as a compact 4x4 grid."""

    print("Greedy policy:")
    for row in policy:
        print(" ".join(row))


def main() -> None:
    """Run training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    q_table, rewards = train(config)
    metrics = evaluate(config, q_table)
    output_dir = save_outputs(config, q_table, rewards, metrics)

    print(f"Run saved to: {output_dir}")
    print(f"Evaluation avg_reward: {metrics['avg_reward']:.4f}")
    print(f"Evaluation success_rate: {metrics['success_rate']:.4f}")

    if config.render_final_policy:
        print_policy(render_policy(q_table))


if __name__ == "__main__":
    main()
