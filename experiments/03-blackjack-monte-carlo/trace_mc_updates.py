"""Trace how first-visit Monte Carlo updates use full-episode returns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TypeAlias


BlackjackState: TypeAlias = tuple[int, int, bool]
ACTION_LABELS = ["Stick", "Hit"]
SCRIPTED_EPISODES: list[list[tuple[BlackjackState, int, float]]] = [
    [
        ((13, 2, False), 1, 0.0),
        ((20, 2, False), 0, 1.0),
    ],
    [
        ((13, 2, False), 1, 0.0),
        ((17, 2, False), 1, -1.0),
    ],
    [
        ((13, 2, False), 1, 0.0),
        ((20, 2, False), 0, 1.0),
    ],
]


@dataclass
class TraceConfig:
    """Configuration for the scripted Monte Carlo trace."""

    episodes: int = 3
    gamma: float = 1.0


def parse_args() -> TraceConfig:
    parser = argparse.ArgumentParser(description="Trace first-visit Monte Carlo updates on scripted Blackjack episodes.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of scripted episodes to replay.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor used when computing returns.")
    args = parser.parse_args()
    max_episodes = min(args.episodes, len(SCRIPTED_EPISODES))
    return TraceConfig(episodes=max_episodes, gamma=args.gamma)


def format_state(state: BlackjackState) -> str:
    """Format a Blackjack state as a readable tuple."""

    player_sum, dealer_card, usable_ace = state
    return f"({player_sum}, dealer={dealer_card}, usable_ace={usable_ace})"


def trace(config: TraceConfig) -> None:
    """Replay scripted episodes and show how sample-average MC updates behave."""

    q_values: dict[tuple[BlackjackState, int], float] = {}
    visit_counts: dict[tuple[BlackjackState, int], int] = {}

    print("Blackjack first-visit Monte Carlo trace")
    print("This script replays a few fixed episodes to show how returns are computed after the episode ends.")
    print()

    for episode_index in range(config.episodes):
        episode = SCRIPTED_EPISODES[episode_index]
        print(f"Episode {episode_index + 1}")
        for step_index, (state, action, reward) in enumerate(episode, start=1):
            print(
                f"  Step {step_index}: state={format_state(state)}, "
                f"action={ACTION_LABELS[action]}, reward={reward:.1f}"
            )

        returns = [0.0] * len(episode)
        discounted_return = 0.0
        for index in range(len(episode) - 1, -1, -1):
            _, _, reward = episode[index]
            discounted_return = reward + config.gamma * discounted_return
            returns[index] = discounted_return

        print("  Returns after the episode finishes:")
        for step_index, episode_return in enumerate(returns, start=1):
            print(f"    G(step {step_index}) = {episode_return:.6f}")

        seen_pairs: set[tuple[BlackjackState, int]] = set()
        for step_index, ((state, action, _), episode_return) in enumerate(zip(episode, returns, strict=True), start=1):
            key = (state, action)
            if key in seen_pairs:
                print(f"    Skip step {step_index} because this state-action pair already appeared earlier.")
                continue
            seen_pairs.add(key)

            old_value = q_values.get(key, 0.0)
            visit_counts[key] = visit_counts.get(key, 0) + 1
            step_size = 1.0 / visit_counts[key]
            new_value = old_value + step_size * (episode_return - old_value)
            q_values[key] = new_value

            print(
                f"    Update {format_state(state)} + {ACTION_LABELS[action]}: "
                f"count={visit_counts[key]}, old={old_value:.6f}, "
                f"G={episode_return:.6f}, new={new_value:.6f}"
            )
        print()

    tracked_key = ((13, 2, False), 1)
    print(
        "Tracked result for state-action "
        f"{format_state(tracked_key[0])} + {ACTION_LABELS[tracked_key[1]]}: "
        f"{q_values.get(tracked_key, 0.0):.6f}"
    )


def main() -> None:
    """Run the CLI entrypoint."""

    config = parse_args()
    trace(config)


if __name__ == "__main__":
    main()
