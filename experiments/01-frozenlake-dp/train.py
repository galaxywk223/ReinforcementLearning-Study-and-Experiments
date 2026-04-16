"""Solve FrozenLake-v1 with dynamic programming baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ACTION_SYMBOLS = ["L", "D", "R", "U"]


@dataclass
class Config:
    """Configuration for a single FrozenLake dynamic-programming run."""

    gamma: float = 0.99
    tolerance: float = 1e-8
    max_iterations: int = 500
    eval_episodes: int = 200
    seed: int = 42
    is_slippery: bool = True
    run_name: str = "dp-reference"
    render_final_policy: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run dynamic-programming baselines on FrozenLake-v1 and save the resulting metrics."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used by both DP solvers.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Convergence tolerance used when stopping iterative Bellman updates.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Hard cap on Bellman sweeps used by policy evaluation and value iteration.",
    )
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of greedy evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment resets.")
    parser.add_argument(
        "--non-slippery",
        action="store_true",
        help="Use the deterministic FrozenLake variant instead of the slippery default.",
    )
    parser.add_argument("--run-name", type=str, default="dp-reference", help="Name of the output subdirectory.")
    parser.add_argument(
        "--render-final-policy",
        action="store_true",
        help="Print the final greedy policy derived from value iteration.",
    )
    args = parser.parse_args()
    return Config(
        gamma=args.gamma,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        is_slippery=not args.non_slippery,
        run_name=args.run_name,
        render_final_policy=args.render_final_policy,
    )


def make_env(is_slippery: bool, seed: int) -> gym.Env:
    """Create a seeded FrozenLake environment."""

    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def get_transition_model(env: gym.Env) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
    """Expose the transition model stored by FrozenLake."""

    return env.unwrapped.P  # type: ignore[attr-defined]


def policy_evaluation(
    policy: np.ndarray,
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, list[float]]:
    """Evaluate a deterministic policy by repeated Bellman sweeps."""

    num_states = len(model)
    values = np.zeros(num_states, dtype=np.float64)
    deltas: list[float] = []

    for _ in range(max_iterations):
        delta = 0.0
        updated = values.copy()

        for state in range(num_states):
            action = int(policy[state])
            expected_value = 0.0
            for probability, next_state, reward, terminated in model[state][action]:
                bootstrap = 0.0 if terminated else values[next_state]
                expected_value += probability * (reward + gamma * bootstrap)
            updated[state] = expected_value
            delta = max(delta, abs(updated[state] - values[state]))

        values = updated
        deltas.append(float(delta))
        if delta < tolerance:
            break

    return values, deltas


def greedy_action_values(
    values: np.ndarray,
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
) -> np.ndarray:
    """Compute Bellman optimality targets for all state-action pairs."""

    num_states = len(model)
    num_actions = len(model[0])
    action_values = np.zeros((num_states, num_actions), dtype=np.float64)

    for state in range(num_states):
        for action in range(num_actions):
            expected_value = 0.0
            for probability, next_state, reward, terminated in model[state][action]:
                bootstrap = 0.0 if terminated else values[next_state]
                expected_value += probability * (reward + gamma * bootstrap)
            action_values[state, action] = expected_value

    return action_values


def improve_policy(
    values: np.ndarray,
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
) -> np.ndarray:
    """Return the greedy policy induced by the current state values."""

    action_values = greedy_action_values(values, model, gamma)
    return np.argmax(action_values, axis=1).astype(np.int64)


def policy_iteration(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """Run policy iteration until the greedy policy becomes stable."""

    num_states = len(model)
    policy = np.zeros(num_states, dtype=np.int64)
    all_deltas: list[float] = []
    improvement_steps = 0

    while improvement_steps < max_iterations:
        values, deltas = policy_evaluation(policy, model, gamma, tolerance, max_iterations)
        all_deltas.extend(deltas)
        improved_policy = improve_policy(values, model, gamma)
        improvement_steps += 1
        if np.array_equal(improved_policy, policy):
            return policy, values, all_deltas, improvement_steps
        policy = improved_policy

    values, deltas = policy_evaluation(policy, model, gamma, tolerance, max_iterations)
    all_deltas.extend(deltas)
    return policy, values, all_deltas, improvement_steps


def value_iteration(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run value iteration until the Bellman optimality updates converge."""

    num_states = len(model)
    values = np.zeros(num_states, dtype=np.float64)
    deltas: list[float] = []

    for _ in range(max_iterations):
        action_values = greedy_action_values(values, model, gamma)
        updated = action_values.max(axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        deltas.append(delta)
        if delta < tolerance:
            break

    policy = improve_policy(values, model, gamma)
    return policy, values, deltas


def evaluate_policy_in_env(config: Config, policy: np.ndarray) -> dict[str, float]:
    """Evaluate a greedy policy in the original environment."""

    env = make_env(config.is_slippery, config.seed + 1)
    rewards: list[float] = []

    for episode_index in range(config.eval_episodes):
        state, _ = env.reset(seed=config.seed + 1 + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = int(policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(np.asarray(rewards) > 0.0)),
    }


def policy_to_grid(policy: np.ndarray, desc: np.ndarray) -> list[list[str]]:
    """Convert a policy vector into a 4x4 grid with map markers."""

    symbols: list[str] = []

    for state, action in enumerate(policy):
        row, column = divmod(state, desc.shape[1])
        cell = desc[row, column].decode("utf-8")
        if cell == "S":
            symbols.append("S")
        elif cell == "H":
            symbols.append("H")
        elif cell == "G":
            symbols.append("G")
        else:
            symbols.append(ACTION_SYMBOLS[int(action)])

    return [symbols[index : index + desc.shape[1]] for index in range(0, len(symbols), desc.shape[1])]


def values_to_grid(values: np.ndarray, desc: np.ndarray) -> list[list[float]]:
    """Convert a value vector into a rounded 4x4 grid."""

    rows = desc.shape[0]
    columns = desc.shape[1]
    return values.reshape(rows, columns).round(6).tolist()


def save_convergence_curve(
    policy_iteration_deltas: list[float],
    value_iteration_deltas: list[float],
    output_path: Path,
) -> None:
    """Save Bellman-update deltas for both DP solvers."""

    plt.figure(figsize=(10, 4.5))
    if policy_iteration_deltas:
        policy_series = np.maximum(np.asarray(policy_iteration_deltas, dtype=np.float64), 1e-12)
        plt.plot(policy_series, linewidth=2, label="Policy iteration evaluation delta")
    if value_iteration_deltas:
        value_series = np.maximum(np.asarray(value_iteration_deltas, dtype=np.float64), 1e-12)
        plt.plot(value_series, linewidth=2, label="Value iteration delta")
    plt.yscale("log")
    plt.title("FrozenLake Dynamic Programming Convergence")
    plt.xlabel("Bellman sweep")
    plt.ylabel("Max state-value change")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_policy_grid(policy: np.ndarray, values: np.ndarray, desc: np.ndarray, output_path: Path) -> None:
    """Render the final greedy policy and values as a compact grid image."""

    grid_policy = policy_to_grid(policy, desc)
    grid_values = np.asarray(values).reshape(desc.shape)

    figure, axis = plt.subplots(figsize=(6.2, 6.0))
    axis.set_xlim(-0.5, desc.shape[1] - 0.5)
    axis.set_ylim(desc.shape[0] - 0.5, -0.5)
    axis.set_xticks(range(desc.shape[1]))
    axis.set_yticks(range(desc.shape[0]))
    axis.grid(color="black", alpha=0.25)
    axis.set_title("FrozenLake Value Iteration Policy Grid")

    for row in range(desc.shape[0]):
        for column in range(desc.shape[1]):
            cell = desc[row, column].decode("utf-8")
            facecolor = "#f7fbff"
            if cell == "H":
                facecolor = "#fee0d2"
            elif cell == "G":
                facecolor = "#e5f5e0"
            elif cell == "S":
                facecolor = "#deebf7"
            axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1.0, 1.0, color=facecolor, zorder=0))
            axis.text(column, row - 0.08, grid_policy[row][column], ha="center", va="center", fontsize=16)
            axis.text(column, row + 0.22, f"{grid_values[row, column]:.3f}", ha="center", va="center", fontsize=9)

    axis.set_xticklabels([])
    axis.set_yticklabels([])
    plt.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_outputs(
    config: Config,
    desc: np.ndarray,
    policy_iteration_policy: np.ndarray,
    policy_iteration_values: np.ndarray,
    policy_iteration_deltas: list[float],
    policy_iteration_metrics: dict[str, float],
    value_iteration_policy: np.ndarray,
    value_iteration_values: np.ndarray,
    value_iteration_deltas: list[float],
    value_iteration_metrics: dict[str, float],
    improvement_steps: int,
) -> Path:
    """Persist plots and summary metadata for a DP run."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_convergence_curve(policy_iteration_deltas, value_iteration_deltas, output_dir / "convergence_curve.png")
    save_policy_grid(value_iteration_policy, value_iteration_values, desc, output_dir / "policy_grid.png")

    summary: dict[str, Any] = {
        "config": asdict(config),
        "train": {
            "policy_iteration_improvement_steps": improvement_steps,
            "policy_iteration_evaluation_sweeps": len(policy_iteration_deltas),
            "value_iteration_sweeps": len(value_iteration_deltas),
            "policy_iteration_final_delta": float(policy_iteration_deltas[-1]) if policy_iteration_deltas else 0.0,
            "value_iteration_final_delta": float(value_iteration_deltas[-1]) if value_iteration_deltas else 0.0,
        },
        "eval": value_iteration_metrics,
        "policy_iteration": {
            "eval": policy_iteration_metrics,
            "policy": policy_to_grid(policy_iteration_policy, desc),
            "state_values": values_to_grid(policy_iteration_values, desc),
        },
        "value_iteration": {
            "policy": policy_to_grid(value_iteration_policy, desc),
            "state_values": values_to_grid(value_iteration_values, desc),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_policy(policy: list[list[str]]) -> None:
    """Print the final greedy policy as a compact grid."""

    print("Greedy policy:")
    for row in policy:
        print(" ".join(row))


def main() -> None:
    """Run policy iteration, value iteration, and artifact generation from the CLI."""

    config = parse_args()
    env = make_env(config.is_slippery, config.seed)
    model = get_transition_model(env)
    desc = env.unwrapped.desc  # type: ignore[attr-defined]

    policy_iteration_policy, policy_iteration_values, policy_iteration_deltas, improvement_steps = policy_iteration(
        model=model,
        gamma=config.gamma,
        tolerance=config.tolerance,
        max_iterations=config.max_iterations,
    )
    value_iteration_policy, value_iteration_values, value_iteration_deltas = value_iteration(
        model=model,
        gamma=config.gamma,
        tolerance=config.tolerance,
        max_iterations=config.max_iterations,
    )
    env.close()

    policy_iteration_metrics = evaluate_policy_in_env(config, policy_iteration_policy)
    value_iteration_metrics = evaluate_policy_in_env(config, value_iteration_policy)
    output_dir = save_outputs(
        config=config,
        desc=desc,
        policy_iteration_policy=policy_iteration_policy,
        policy_iteration_values=policy_iteration_values,
        policy_iteration_deltas=policy_iteration_deltas,
        policy_iteration_metrics=policy_iteration_metrics,
        value_iteration_policy=value_iteration_policy,
        value_iteration_values=value_iteration_values,
        value_iteration_deltas=value_iteration_deltas,
        value_iteration_metrics=value_iteration_metrics,
        improvement_steps=improvement_steps,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Policy iteration avg_reward: {policy_iteration_metrics['avg_reward']:.4f}")
    print(f"Policy iteration success_rate: {policy_iteration_metrics['success_rate']:.4f}")
    print(f"Value iteration avg_reward: {value_iteration_metrics['avg_reward']:.4f}")
    print(f"Value iteration success_rate: {value_iteration_metrics['success_rate']:.4f}")

    if config.render_final_policy:
        print_policy(policy_to_grid(value_iteration_policy, desc))


if __name__ == "__main__":
    main()
