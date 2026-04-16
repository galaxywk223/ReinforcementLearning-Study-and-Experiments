"""Print how SAC builds its soft Q targets."""

from __future__ import annotations

import argparse
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the target-value formula used by soft actor-critic."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in soft Q targets.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy temperature used in the soft value term.")
    return parser.parse_args()


def main() -> None:
    """Print one toy SAC target batch."""

    args = parse_args()
    rewards = np.asarray([-1.2, -0.7, -2.0, -0.3], dtype=np.float64)
    min_target_q = np.asarray([3.5, 2.0, 4.1, 1.4], dtype=np.float64)
    log_probs = np.asarray([-0.8, -0.2, -1.1, -0.5], dtype=np.float64)
    dones = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    targets = rewards + args.gamma * (1.0 - dones) * (min_target_q - args.alpha * log_probs)

    print(f"{'idx':>3} {'reward':>8} {'min_q':>8} {'log_pi':>8} {'done':>6} {'target':>10}")
    for index in range(len(rewards)):
        print(
            f"{index:>3} {rewards[index]:>8.3f} {min_target_q[index]:>8.3f} "
            f"{log_probs[index]:>8.3f} {dones[index]:>6.1f} {targets[index]:>10.3f}"
        )


if __name__ == "__main__":
    main()
