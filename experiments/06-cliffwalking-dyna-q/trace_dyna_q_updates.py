"""Trace one real Dyna-Q update followed by planning replays."""

from __future__ import annotations

import argparse
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace one real Dyna-Q transition and the subsequent planning updates."
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate for Q-value updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in the Bellman target.")
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=5,
        help="Number of planning updates sampled after the real environment step.",
    )
    return parser.parse_args()


def q_learning_target(q_table: np.ndarray, reward: float, next_state: int, terminated: bool, gamma: float) -> float:
    """Construct the one-step Q-learning target."""

    if terminated:
        return reward
    return reward + gamma * float(np.max(q_table[next_state]))


def main() -> None:
    """Print one real transition followed by repeated model updates."""

    args = parse_args()
    q_table = np.zeros((48, 4), dtype=np.float64)

    state = 36
    action = 1
    reward = -100.0
    next_state = 36
    terminated = False
    model = {(state, action): (reward, next_state, terminated)}

    td_target = q_learning_target(q_table, reward, next_state, terminated, args.gamma)
    q_table[state, action] += args.alpha * (td_target - q_table[state, action])
    print(
        "Real step: "
        f"(s={state}, a=R, r={reward:.1f}, s'={next_state}) -> "
        f"target={td_target:.4f}, updated_q={q_table[state, action]:.4f}"
    )

    for planning_index in range(1, args.planning_steps + 1):
        sampled_reward, sampled_next_state, sampled_done = model[(state, action)]
        planned_target = q_learning_target(q_table, sampled_reward, sampled_next_state, sampled_done, args.gamma)
        q_table[state, action] += args.alpha * (planned_target - q_table[state, action])
        print(
            f"Planning step {planning_index}: "
            f"target={planned_target:.4f}, q(36, R)={q_table[state, action]:.4f}"
        )


if __name__ == "__main__":
    main()
