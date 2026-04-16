"""Trace how Q-values change along a fixed successful FrozenLake path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


ACTION_SYMBOLS = ["L", "D", "R", "U"]
# 一条安全到终点的固定路径：
# S(0,0) -> (1,0) -> (2,0) -> (2,1) -> (3,1) -> (3,2) -> G(3,3)
SCRIPTED_ACTIONS = [1, 1, 2, 1, 2, 2]


@dataclass
class TraceConfig:
    """Configuration for the fixed-path Q-value trace."""

    episodes: int = 6
    alpha: float = 0.1
    gamma: float = 0.99
    seed: int = 42


def parse_args() -> TraceConfig:
    parser = argparse.ArgumentParser(description="Trace how tabular Q-learning updates propagate on FrozenLake.")
    parser.add_argument("--episodes", type=int, default=6, help="Number of scripted successful episodes to replay.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate used in the trace.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in the trace.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment reset.")
    args = parser.parse_args()
    return TraceConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )


def make_env(seed: int) -> gym.Env:
    """Create a deterministic FrozenLake environment for the trace."""

    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset(seed=seed)
    return env


def state_to_row_col(state: int, width: int = 4) -> tuple[int, int]:
    """Convert a flattened state index into row and column coordinates."""

    return divmod(state, width)


def format_state(state: int) -> str:
    """Format a state index as 'index(row,col)'."""

    row, col = state_to_row_col(state)
    return f"{state}({row},{col})"


def print_header() -> None:
    """Print the environment layout and scripted action sequence."""

    print("FrozenLake fixed path trace")
    print("Map:")
    print("S F F F")
    print("F H F H")
    print("F F F H")
    print("H F F G")
    print("Scripted actions:", " -> ".join(ACTION_SYMBOLS[action] for action in SCRIPTED_ACTIONS))
    print()


def print_path_values(q_table: np.ndarray) -> None:
    """Print the tracked Q-values along the fixed successful path."""

    tracked_pairs = [
        (0, 1, "Q(0, D)"),
        (4, 1, "Q(4, D)"),
        (8, 2, "Q(8, R)"),
        (9, 1, "Q(9, D)"),
        (13, 2, "Q(13, R)"),
        (14, 2, "Q(14, R)"),
    ]
    print("Tracked Q-values on the successful path:")
    for state, action, label in tracked_pairs:
        print(f"  {label:<9} = {q_table[state, action]:.6f}")
    print()


def trace(config: TraceConfig) -> None:
    """Replay a fixed successful path and show each Q-value update."""

    env = make_env(config.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

    print_header()
    print("Initial Q-table values are all zero.")
    print_path_values(q_table)

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        print(f"Episode {episode}")

        for step_index, action in enumerate(SCRIPTED_ACTIONS, start=1):
            next_state, reward, terminated, truncated, _ = env.step(action)
            old_q = q_table[state, action]
            next_best = float(np.max(q_table[next_state])) if not terminated else 0.0
            td_target = reward + config.gamma * next_best
            td_error = td_target - old_q
            new_q = old_q + config.alpha * td_error
            q_table[state, action] = new_q

            print(
                f"  Step {step_index}: "
                f"{format_state(state)} --{ACTION_SYMBOLS[action]}--> {format_state(next_state)} | "
                f"reward={reward:.1f}, next_best={next_best:.6f}, "
                f"target={td_target:.6f}, old={old_q:.6f}, new={new_q:.6f}"
            )

            state = next_state
            if terminated or truncated:
                break

        print_path_values(q_table)

    env.close()


def main() -> None:
    """Run the CLI entrypoint."""

    config = parse_args()
    trace(config)


if __name__ == "__main__":
    main()
