"""Train a tabular Dyna-Q agent on CliffWalking-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ACTION_SYMBOLS = ["U", "R", "D", "L"]
GRID_HEIGHT = 4
GRID_WIDTH = 12
START_STATE = 36
GOAL_STATE = 47
CLIFF_STATES = set(range(37, 47))


@dataclass
class Config:
    """Configuration for a single CliffWalking Dyna-Q run."""

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
    run_name: str = "dyna-q-baseline"
    render_final_policy: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a tabular Dyna-Q agent on CliffWalking-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon-start", type=float, default=0.1, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=1.0, help="Per-episode epsilon decay factor.")
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=10,
        help="Number of model rollouts sampled after each real environment step.",
    )
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Hard cap on steps per episode to avoid endless wandering in CliffWalking.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment and action sampling.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=40,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument("--run-name", type=str, default="dyna-q-baseline", help="Name of the output subdirectory.")
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
        planning_steps=args.planning_steps,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        render_final_policy=args.render_final_policy,
    )


def make_env(seed: int) -> gym.Env:
    """Create a seeded CliffWalking environment."""

    env = gym.make("CliffWalking-v1")
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


def q_learning_target(q_table: np.ndarray, reward: float, next_state: int, terminated: bool, gamma: float) -> float:
    """Construct the one-step Q-learning target."""

    if terminated:
        return reward
    return reward + gamma * float(np.max(q_table[next_state]))


def run_training(
    config: Config,
    planning_steps: int,
) -> tuple[np.ndarray, list[float], dict[tuple[int, int], tuple[float, int, bool]], dict[str, float]]:
    """Train either plain Q-learning or Dyna-Q, depending on the planning budget."""

    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    model: dict[tuple[int, int], tuple[float, int, bool]] = {}
    rewards: list[float] = []
    epsilon = config.epsilon_start

    for _ in range(config.episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < config.max_steps_per_episode:
            action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            done = bool(terminated or truncated)
            td_target = q_learning_target(q_table, reward, next_state, done, config.gamma)
            q_table[state, action] += config.alpha * (td_target - q_table[state, action])
            model[(state, action)] = (float(reward), int(next_state), done)

            if model and planning_steps > 0:
                seen_pairs = list(model.keys())
                for _ in range(planning_steps):
                    sampled_state, sampled_action = seen_pairs[int(rng.integers(len(seen_pairs)))]
                    sampled_reward, sampled_next_state, sampled_done = model[(sampled_state, sampled_action)]
                    planned_target = q_learning_target(
                        q_table=q_table,
                        reward=sampled_reward,
                        next_state=sampled_next_state,
                        terminated=sampled_done,
                        gamma=config.gamma,
                    )
                    q_table[sampled_state, sampled_action] += config.alpha * (
                        planned_target - q_table[sampled_state, sampled_action]
                    )

            state = next_state

        rewards.append(total_reward)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    env.close()
    train_stats = {
        "episodes": config.episodes,
        "avg_reward_last_100": float(np.mean(rewards[-100:])) if rewards else 0.0,
        "best_episode_reward": float(np.max(rewards)) if rewards else 0.0,
        "final_epsilon": float(epsilon),
        "learned_model_size": float(len(model)),
        "planning_steps_per_real_step": float(planning_steps),
    }
    return q_table, rewards, model, train_stats


def train(config: Config) -> tuple[np.ndarray, list[float], dict[tuple[int, int], tuple[float, int, bool]], dict[str, float]]:
    """Train Dyna-Q and return the learned Q-table, rewards, learned model, and stats."""

    return run_training(config, planning_steps=config.planning_steps)


def evaluate(config: Config, q_table: np.ndarray) -> dict[str, float]:
    """Evaluate the greedy policy induced by the learned Q-table."""

    env = make_env(config.seed + 1)
    rewards: list[float] = []
    steps_to_goal: list[int] = []
    cliff_falls: list[int] = []

    for episode_index in range(config.eval_episodes):
        state, _ = env.reset(seed=config.seed + 1 + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        falls = 0

        while not (terminated or truncated) and steps < config.max_steps_per_episode:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if reward <= -100:
                falls += 1

        rewards.append(total_reward)
        steps_to_goal.append(steps)
        cliff_falls.append(falls)

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_steps_to_goal": float(np.mean(steps_to_goal)),
        "avg_cliff_falls": float(np.mean(cliff_falls)),
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
    plt.title("CliffWalking Dyna-Q Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(rewards))})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def render_policy(q_table: np.ndarray) -> list[list[str]]:
    """Convert the greedy policy into a 4x12 grid of action symbols."""

    best_actions = np.argmax(q_table, axis=1)
    symbols: list[str] = []

    for state in range(len(best_actions)):
        if state == START_STATE:
            symbols.append("S")
        elif state in CLIFF_STATES:
            symbols.append("C")
        elif state == GOAL_STATE:
            symbols.append("G")
        else:
            symbols.append(ACTION_SYMBOLS[int(best_actions[state])])

    return [symbols[index : index + GRID_WIDTH] for index in range(0, len(symbols), GRID_WIDTH)]


def save_outputs(
    config: Config,
    q_table: np.ndarray,
    rewards: list[float],
    train_stats: dict[str, float],
    metrics: dict[str, float],
) -> Path:
    """Persist plots and summary metadata for a training run."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_reward_curve(rewards, config.moving_avg_window, output_dir / "reward_curve.png")
    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": metrics,
        "policy": render_policy(q_table),
        "q_table": q_table.round(6).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_policy(policy: list[list[str]]) -> None:
    """Print the greedy policy as a compact 4x12 grid."""

    print("Greedy policy:")
    for row in policy:
        print(" ".join(row))


def main() -> None:
    """Run Dyna-Q training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    q_table, rewards, _, train_stats = train(config)
    metrics = evaluate(config, q_table)
    output_dir = save_outputs(config, q_table, rewards, train_stats, metrics)

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_100: {train_stats['avg_reward_last_100']:.4f}")
    print(f"Evaluation avg_reward: {metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_steps_to_goal: {metrics['avg_steps_to_goal']:.4f}")
    print(f"Evaluation avg_cliff_falls: {metrics['avg_cliff_falls']:.4f}")

    if config.render_final_policy:
        print_policy(render_policy(q_table))


if __name__ == "__main__":
    main()
