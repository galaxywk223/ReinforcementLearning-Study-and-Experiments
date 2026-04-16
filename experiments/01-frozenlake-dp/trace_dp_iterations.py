"""Print the first few value-iteration sweeps on FrozenLake-v1."""

from __future__ import annotations

import argparse
import numpy as np

from train import ACTION_SYMBOLS, get_transition_model, greedy_action_values, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the first few value-iteration sweeps on FrozenLake-v1."
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of Bellman sweeps to print.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in the Bellman target.")
    parser.add_argument(
        "--non-slippery",
        action="store_true",
        help="Use the deterministic FrozenLake variant instead of the slippery default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when creating the environment.")
    return parser.parse_args()


def format_grid(values: np.ndarray) -> str:
    """Format a 4x4 value grid for CLI output."""

    rows = values.reshape(4, 4)
    return "\n".join("  " + " ".join(f"{cell:6.3f}" for cell in row) for row in rows)


def format_policy(values: np.ndarray, model: dict[int, dict[int, list[tuple[float, int, float, bool]]]], gamma: float) -> str:
    """Format the greedy policy induced by the current value grid."""

    action_values = greedy_action_values(values, model, gamma)
    greedy_actions = np.argmax(action_values, axis=1)
    rows = greedy_actions.reshape(4, 4)
    return "\n".join("  " + " ".join(ACTION_SYMBOLS[int(action)] for action in row) for row in rows)


def main() -> None:
    """Print the first few value-iteration sweeps."""

    args = parse_args()
    env = make_env(is_slippery=not args.non_slippery, seed=args.seed)
    model = get_transition_model(env)
    values = np.zeros(env.observation_space.n, dtype=np.float64)

    for iteration in range(1, args.iterations + 1):
        action_values = greedy_action_values(values, model, args.gamma)
        updated = action_values.max(axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        print(f"Iteration {iteration}: delta={delta:.6e}")
        print("Values:")
        print(format_grid(values))
        print("Greedy policy:")
        print(format_policy(values, model, args.gamma))
        print()

    env.close()


if __name__ == "__main__":
    main()
