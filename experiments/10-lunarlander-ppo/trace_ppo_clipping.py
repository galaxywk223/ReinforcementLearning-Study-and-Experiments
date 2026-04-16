"""Print how PPO clipping changes the surrogate objective."""

from __future__ import annotations

import argparse
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace how PPO's clipped ratio modifies the surrogate objective."
    )
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO ratio clipping threshold.")
    return parser.parse_args()


def main() -> None:
    """Print one toy PPO minibatch before and after clipping."""

    args = parse_args()
    advantages = np.asarray([1.2, 0.6, -0.4, -1.0], dtype=np.float64)
    old_log_probs = np.asarray([-0.8, -1.1, -0.5, -0.9], dtype=np.float64)
    new_log_probs = np.asarray([-0.4, -1.0, -0.8, -1.3], dtype=np.float64)
    ratios = np.exp(new_log_probs - old_log_probs)
    clipped_ratios = np.clip(ratios, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
    unclipped = ratios * advantages
    clipped = clipped_ratios * advantages
    surrogate = np.minimum(unclipped, clipped)

    print(f"{'idx':>3} {'adv':>8} {'ratio':>8} {'unclipped':>11} {'clipped':>11} {'used':>11}")
    for index in range(len(advantages)):
        print(
            f"{index:>3} {advantages[index]:>8.3f} {ratios[index]:>8.3f} "
            f"{unclipped[index]:>11.3f} {clipped[index]:>11.3f} {surrogate[index]:>11.3f}"
        )


if __name__ == "__main__":
    main()
