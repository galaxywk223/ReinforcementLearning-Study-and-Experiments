"""Compare tabular SARSA and tabular Q-learning on CliffWalking-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ACTION_SYMBOLS = ["U", "R", "D", "L"]
GRID_WIDTH = 12
START_STATE = 36
GOAL_STATE = 47
CLIFF_STATES = set(range(37, 47))


@dataclass
class CompareConfig:
    """Configuration for comparing SARSA and Q-learning."""

    episodes: int = 800
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon: float = 0.1
    eval_episodes: int = 200
    max_steps_per_episode: int = 500
    seed: int = 42
    moving_avg_window: int = 50
    run_name: str = "sarsa-vs-q-learning"


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(description="Compare tabular SARSA and Q-learning on CliffWalking-v1.")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes per algorithm.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed exploration rate used during training.")
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
        default=50,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="sarsa-vs-q-learning",
        help="Name of the comparison output subdirectory.",
    )
    args = parser.parse_args()
    return CompareConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
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


def train_sarsa(config: CompareConfig) -> tuple[np.ndarray, list[float]]:
    """Train SARSA and return the learned Q-table and raw episode rewards."""

    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rewards: list[float] = []

    for _ in range(config.episodes):
        state, _ = env.reset()
        action = epsilon_greedy_action(q_table, state, config.epsilon, env.action_space, rng)
        terminated = False
        truncated = False
        total_reward = 0.0

        steps = 0
        while not (terminated or truncated) and steps < config.max_steps_per_episode:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                td_target = reward
                next_action = None
            else:
                next_action = epsilon_greedy_action(q_table, next_state, config.epsilon, env.action_space, rng)
                td_target = reward + config.gamma * q_table[next_state, next_action]

            td_error = td_target - q_table[state, action]
            q_table[state, action] += config.alpha * td_error

            state = next_state
            if next_action is not None:
                action = next_action

        rewards.append(total_reward)

    env.close()
    return q_table, rewards


def train_q_learning(config: CompareConfig) -> tuple[np.ndarray, list[float]]:
    """Train Q-learning and return the learned Q-table and raw episode rewards."""

    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rewards: list[float] = []

    for _ in range(config.episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        steps = 0
        while not (terminated or truncated) and steps < config.max_steps_per_episode:
            action = epsilon_greedy_action(q_table, state, config.epsilon, env.action_space, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            td_target = reward
            if not (terminated or truncated):
                td_target += config.gamma * np.max(q_table[next_state])

            td_error = td_target - q_table[state, action]
            q_table[state, action] += config.alpha * td_error
            state = next_state

        rewards.append(total_reward)

    env.close()
    return q_table, rewards


def evaluate(config: CompareConfig, q_table: np.ndarray) -> dict[str, float]:
    """Evaluate a greedy policy induced by a Q-table."""

    env = make_env(config.seed + 1)
    rewards: list[float] = []
    steps_to_goal: list[int] = []
    cliff_falls: list[int] = []

    for _ in range(config.eval_episodes):
        state, _ = env.reset()
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


def state_to_row_col(state: int) -> tuple[int, int]:
    """Convert a flattened state index into row and column coordinates."""

    return divmod(state, GRID_WIDTH)


def format_state(state: int) -> str:
    """Format a state index as 'index(row,col)'."""

    row, col = state_to_row_col(state)
    return f"{state}({row},{col})"


def greedy_rollout(config: CompareConfig, q_table: np.ndarray) -> dict[str, object]:
    """Roll out the greedy policy from the start state and record the realized path."""

    env = make_env(config.seed + 2)
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    cliff_falls = 0
    visited_states = [format_state(state)]
    transitions: list[str] = []
    steps = 0

    while not (terminated or truncated) and steps < config.max_steps_per_episode:
        action = int(np.argmax(q_table[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        transitions.append(
            f"{format_state(state)} --{ACTION_SYMBOLS[action]}--> {format_state(next_state)} (reward={reward:.1f})"
        )
        visited_states.append(format_state(next_state))
        total_reward += reward
        steps += 1

        if reward <= -100:
            cliff_falls += 1

        state = next_state

    env.close()
    return {
        "visited_states": visited_states,
        "transitions": transitions,
        "steps": steps,
        "total_reward": float(total_reward),
        "cliff_falls": cliff_falls,
        "terminated": terminated,
        "truncated": truncated or (steps >= config.max_steps_per_episode and not terminated),
    }


def save_comparison_curve(
    sarsa_rewards: list[float], q_learning_rewards: list[float], window: int, output_path: Path
) -> None:
    """Save the smoothed reward curves for both algorithms."""

    sarsa_smoothed = moving_average(sarsa_rewards, window)
    q_learning_smoothed = moving_average(q_learning_rewards, window)

    plt.figure(figsize=(10, 4.5))
    plt.plot(sarsa_smoothed, linewidth=2, label="SARSA")
    plt.plot(q_learning_smoothed, linewidth=2, label="Q-Learning")
    plt.title("CliffWalking: SARSA vs Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(sarsa_rewards), len(q_learning_rewards))})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: CompareConfig,
    sarsa_q: np.ndarray,
    sarsa_rewards: list[float],
    sarsa_metrics: dict[str, float],
    q_learning_q: np.ndarray,
    q_learning_rewards: list[float],
    q_learning_metrics: dict[str, float],
) -> Path:
    """Persist comparison plots and summary metadata."""

    output_dir = Path(__file__).resolve().parent / "outputs" / "comparisons" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_curve(
        sarsa_rewards=sarsa_rewards,
        q_learning_rewards=q_learning_rewards,
        window=config.moving_avg_window,
        output_path=output_dir / "comparison_reward_curve.png",
    )

    summary = {
        "config": asdict(config),
        "sarsa": {
            "train": {
                "episodes": config.episodes,
                "avg_reward_last_100": float(np.mean(sarsa_rewards[-100:])) if sarsa_rewards else 0.0,
                "best_episode_reward": float(np.max(sarsa_rewards)) if sarsa_rewards else 0.0,
            },
            "eval": sarsa_metrics,
            "policy": render_policy(sarsa_q),
            "greedy_rollout": greedy_rollout(config, sarsa_q),
            "q_table": sarsa_q.round(6).tolist(),
        },
        "q_learning": {
            "train": {
                "episodes": config.episodes,
                "avg_reward_last_100": float(np.mean(q_learning_rewards[-100:])) if q_learning_rewards else 0.0,
                "best_episode_reward": float(np.max(q_learning_rewards)) if q_learning_rewards else 0.0,
            },
            "eval": q_learning_metrics,
            "policy": render_policy(q_learning_q),
            "greedy_rollout": greedy_rollout(config, q_learning_q),
            "q_table": q_learning_q.round(6).tolist(),
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
    sarsa_q, sarsa_rewards = train_sarsa(config)
    q_learning_q, q_learning_rewards = train_q_learning(config)
    sarsa_metrics = evaluate(config, sarsa_q)
    q_learning_metrics = evaluate(config, q_learning_q)
    output_dir = save_outputs(
        config=config,
        sarsa_q=sarsa_q,
        sarsa_rewards=sarsa_rewards,
        sarsa_metrics=sarsa_metrics,
        q_learning_q=q_learning_q,
        q_learning_rewards=q_learning_rewards,
        q_learning_metrics=q_learning_metrics,
    )

    print(f"Comparison saved to: {output_dir}")
    print(
        "SARSA eval: "
        f"avg_reward={sarsa_metrics['avg_reward']:.4f}, "
        f"avg_steps_to_goal={sarsa_metrics['avg_steps_to_goal']:.4f}, "
        f"avg_cliff_falls={sarsa_metrics['avg_cliff_falls']:.4f}"
    )
    print(
        "Q-Learning eval: "
        f"avg_reward={q_learning_metrics['avg_reward']:.4f}, "
        f"avg_steps_to_goal={q_learning_metrics['avg_steps_to_goal']:.4f}, "
        f"avg_cliff_falls={q_learning_metrics['avg_cliff_falls']:.4f}"
    )
    print()

    print_policy("SARSA", render_policy(sarsa_q))
    print_policy("Q-Learning", render_policy(q_learning_q))
    print_greedy_rollout("SARSA", greedy_rollout(config, sarsa_q))
    print_greedy_rollout("Q-Learning", greedy_rollout(config, q_learning_q))


if __name__ == "__main__":
    main()
