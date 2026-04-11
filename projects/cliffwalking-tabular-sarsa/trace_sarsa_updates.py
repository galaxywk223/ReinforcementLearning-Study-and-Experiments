"""Trace how SARSA updates use the actual next action on a fixed CliffWalking path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


ACTION_SYMBOLS = ["U", "R", "D", "L"]
SCRIPTED_ACTIONS = [0] + [1] * 11 + [2]


@dataclass
class TraceConfig:
    """Configuration for the fixed-path SARSA trace."""

    episodes: int = 2
    alpha: float = 0.5
    gamma: float = 0.99
    seed: int = 42


def parse_args() -> TraceConfig:
    parser = argparse.ArgumentParser(description="Trace how tabular SARSA updates propagate on CliffWalking.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of scripted successful episodes to replay.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate used in the trace.")
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
    """Create a deterministic CliffWalking environment for the trace."""

    env = gym.make("CliffWalking-v1")
    env.reset(seed=seed)
    return env


def state_to_row_col(state: int, width: int = 12) -> tuple[int, int]:
    """Convert a flattened state index into row and column coordinates."""

    return divmod(state, width)


def format_state(state: int) -> str:
    """Format a state index as 'index(row,col)'."""

    row, col = state_to_row_col(state)
    return f"{state}({row},{col})"


def print_header() -> None:
    """Print the environment layout and scripted action sequence."""

    print("CliffWalking fixed path SARSA trace")
    print("Map:")
    print(". . . . . . . . . . . .")
    print(". . . . . . . . . . . .")
    print(". . . . . . . . . . . .")
    print("S C C C C C C C C C C G")
    print("Scripted actions:", " -> ".join(ACTION_SYMBOLS[action] for action in SCRIPTED_ACTIONS))
    print()


def print_path_values(q_table: np.ndarray) -> None:
    """Print tracked Q-values along the scripted safe path."""

    tracked_pairs = [
        (36, 0, "Q(36, U)"),
        (24, 1, "Q(24, R)"),
        (25, 1, "Q(25, R)"),
        (34, 1, "Q(34, R)"),
        (35, 2, "Q(35, D)"),
    ]
    print("Tracked Q-values on the safe path:")
    for state, action, label in tracked_pairs:
        print(f"  {label:<9} = {q_table[state, action]:.6f}")
    print()


def trace(config: TraceConfig) -> None:
    """Replay a fixed safe path and show each SARSA update."""

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

            if terminated or truncated:
                next_action = None
                next_q = 0.0
                td_target = reward
            else:
                next_action = SCRIPTED_ACTIONS[step_index]
                next_q = float(q_table[next_state, next_action])
                td_target = reward + config.gamma * next_q

            td_error = td_target - old_q
            new_q = old_q + config.alpha * td_error
            q_table[state, action] = new_q

            next_action_symbol = ACTION_SYMBOLS[next_action] if next_action is not None else "terminal"
            print(
                f"  Step {step_index}: "
                f"{format_state(state)} --{ACTION_SYMBOLS[action]}--> {format_state(next_state)} | "
                f"reward={reward:.1f}, next_action={next_action_symbol}, next_q={next_q:.6f}, "
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
