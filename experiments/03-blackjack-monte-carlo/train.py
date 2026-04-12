"""Train a first-visit Monte Carlo control agent on Blackjack-v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


BlackjackState: TypeAlias = tuple[int, int, bool]
ACTION_LABELS = ["Stick", "Hit"]
PLAYER_SUMS = list(range(21, 11, -1))
DEALER_SHOWING = list(range(1, 11))


@dataclass
class Config:
    """Configuration for a single Blackjack Monte Carlo training run."""

    episodes: int = 200000
    gamma: float = 1.0
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99999
    eval_episodes: int = 10000
    seed: int = 42
    moving_avg_window: int = 5000
    run_name: str = "monte-carlo-baseline"
    render_final_policy: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a first-visit Monte Carlo control agent on Blackjack-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=200000, help="Number of training episodes.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor used when computing returns.")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=0.99999, help="Per-episode epsilon decay factor.")
    parser.add_argument("--eval-episodes", type=int, default=10000, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment and action sampling.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=5000,
        help="Window size used when smoothing the reward curve.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="monte-carlo-baseline",
        help="Name of the output subdirectory.",
    )
    parser.add_argument(
        "--render-final-policy",
        action="store_true",
        help="Print the final greedy policy after training.",
    )
    args = parser.parse_args()
    return Config(
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        render_final_policy=args.render_final_policy,
    )


def make_env(seed: int) -> gym.Env:
    """Create a seeded Blackjack environment."""

    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def state_to_index(state: BlackjackState) -> tuple[int, int, int]:
    """Convert a Blackjack observation into Q-table indices."""

    player_sum, dealer_card, usable_ace = state
    return int(player_sum), int(dealer_card), int(bool(usable_ace))


def greedy_action(q_table: np.ndarray, state: BlackjackState, rng: np.random.Generator | None = None) -> int:
    """Choose a greedy action, optionally breaking ties at random."""

    action_values = q_table[state_to_index(state)]
    best_actions = np.flatnonzero(np.isclose(action_values, np.max(action_values)))
    if rng is not None and len(best_actions) > 1:
        return int(rng.choice(best_actions))
    return int(best_actions[0])


def epsilon_greedy_action(
    q_table: np.ndarray, state: BlackjackState, epsilon: float, rng: np.random.Generator, action_space: gym.Space
) -> int:
    """Choose an action using epsilon-greedy exploration."""

    if rng.random() < epsilon:
        return int(action_space.sample())
    return greedy_action(q_table, state, rng=rng)


def train(config: Config) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train the agent and return the learned Q-table, visit counts, and raw rewards."""

    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    q_table = np.zeros((32, 11, 2, env.action_space.n), dtype=np.float64)
    visit_counts = np.zeros_like(q_table, dtype=np.int32)
    episode_rewards: list[float] = []
    epsilon = config.epsilon_start

    for _ in range(config.episodes):
        state, _ = env.reset()
        state = tuple(state)
        terminated = False
        truncated = False
        total_reward = 0.0
        episode: list[tuple[BlackjackState, int, float]] = []

        while not (terminated or truncated):
            action = epsilon_greedy_action(q_table, state, epsilon, rng, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, float(reward)))
            total_reward += reward
            state = tuple(next_state)

        returns = [0.0] * len(episode)
        discounted_return = 0.0
        for index in range(len(episode) - 1, -1, -1):
            _, _, reward = episode[index]
            discounted_return = reward + config.gamma * discounted_return
            returns[index] = discounted_return

        seen_pairs: set[tuple[BlackjackState, int]] = set()
        for index, (episode_state, action, _) in enumerate(episode):
            key = (episode_state, action)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            table_index = state_to_index(episode_state) + (action,)
            visit_counts[table_index] += 1
            step_size = 1.0 / visit_counts[table_index]
            q_table[table_index] += step_size * (returns[index] - q_table[table_index])

        episode_rewards.append(total_reward)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    env.close()
    return q_table, visit_counts, episode_rewards


def evaluate(config: Config, q_table: np.ndarray) -> dict[str, float]:
    """Evaluate the greedy policy induced by the learned Q-table."""

    env = make_env(config.seed + 1)
    rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        state = tuple(state)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = greedy_action(q_table, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = tuple(next_state)

        rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    reward_array = np.asarray(rewards, dtype=np.float64)
    return {
        "avg_reward": float(np.mean(reward_array)),
        "win_rate": float(np.mean(reward_array > 0.0)),
        "draw_rate": float(np.mean(reward_array == 0.0)),
        "loss_rate": float(np.mean(reward_array < 0.0)),
        "avg_episode_length": float(np.mean(episode_lengths)),
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
    plt.title("Blackjack First-Visit Monte Carlo Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={min(window, len(rewards))})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def policy_grids(q_table: np.ndarray) -> dict[str, np.ndarray]:
    """Render greedy actions as 10x10 policy grids."""

    grids: dict[str, np.ndarray] = {}

    for ace_flag, label in ((0, "no_usable_ace"), (1, "usable_ace")):
        grid = np.zeros((len(PLAYER_SUMS), len(DEALER_SHOWING)), dtype=np.int32)
        for row, player_sum in enumerate(PLAYER_SUMS):
            for col, dealer_card in enumerate(DEALER_SHOWING):
                grid[row, col] = greedy_action(q_table, (player_sum, dealer_card, bool(ace_flag)))
        grids[label] = grid

    return grids


def value_grids(q_table: np.ndarray) -> dict[str, np.ndarray]:
    """Render greedy state values as 10x10 value grids."""

    grids: dict[str, np.ndarray] = {}

    for ace_flag, label in ((0, "no_usable_ace"), (1, "usable_ace")):
        grid = np.zeros((len(PLAYER_SUMS), len(DEALER_SHOWING)), dtype=np.float64)
        for row, player_sum in enumerate(PLAYER_SUMS):
            for col, dealer_card in enumerate(DEALER_SHOWING):
                grid[row, col] = float(np.max(q_table[player_sum, dealer_card, ace_flag]))
        grids[label] = grid

    return grids


def save_policy_plots(q_table: np.ndarray, output_dir: Path) -> None:
    """Save greedy policy heatmaps for states with and without a usable ace."""

    grids = policy_grids(q_table)
    cmap = ListedColormap(["#f2efe8", "#274c77"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for axis, (label, title) in zip(
        axes,
        (("no_usable_ace", "No Usable Ace"), ("usable_ace", "Usable Ace")),
        strict=True,
    ):
        image = axis.imshow(grids[label], cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axis.set_title(f"Greedy Policy: {title}")
        axis.set_xticks(range(len(DEALER_SHOWING)), labels=DEALER_SHOWING)
        axis.set_yticks(range(len(PLAYER_SUMS)), labels=PLAYER_SUMS)
        axis.set_xlabel("Dealer Showing")
        axis.set_ylabel("Player Sum")

        for row in range(len(PLAYER_SUMS)):
            for col in range(len(DEALER_SHOWING)):
                action = ACTION_LABELS[int(grids[label][row, col])]
                text_color = "white" if grids[label][row, col] == 1 else "#1f1f1f"
                axis.text(col, row, action[0], ha="center", va="center", color=text_color, fontsize=9)

    colorbar = figure.colorbar(image, ax=axes, ticks=[0, 1], shrink=0.92)
    colorbar.ax.set_yticklabels(ACTION_LABELS)
    figure.savefig(output_dir / "policy_heatmaps.png", dpi=160)
    plt.close(figure)


def save_value_plots(q_table: np.ndarray, output_dir: Path) -> None:
    """Save greedy state-value heatmaps for states with and without a usable ace."""

    grids = value_grids(q_table)
    value_min = float(min(np.min(grid) for grid in grids.values()))
    value_max = float(max(np.max(grid) for grid in grids.values()))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for axis, (label, title) in zip(
        axes,
        (("no_usable_ace", "No Usable Ace"), ("usable_ace", "Usable Ace")),
        strict=True,
    ):
        image = axis.imshow(
            grids[label],
            cmap="coolwarm",
            vmin=value_min,
            vmax=value_max,
            aspect="auto",
        )
        axis.set_title(f"Greedy State Value: {title}")
        axis.set_xticks(range(len(DEALER_SHOWING)), labels=DEALER_SHOWING)
        axis.set_yticks(range(len(PLAYER_SUMS)), labels=PLAYER_SUMS)
        axis.set_xlabel("Dealer Showing")
        axis.set_ylabel("Player Sum")

    figure.colorbar(image, ax=axes, shrink=0.92)
    figure.savefig(output_dir / "value_heatmaps.png", dpi=160)
    plt.close(figure)


def policy_tables(q_table: np.ndarray) -> dict[str, list[list[str]]]:
    """Convert greedy actions into labeled policy tables."""

    grids = policy_grids(q_table)
    tables: dict[str, list[list[str]]] = {}

    for label, grid in grids.items():
        tables[label] = [[ACTION_LABELS[int(action)] for action in row] for row in grid]

    return tables


def value_tables(q_table: np.ndarray) -> dict[str, list[list[float]]]:
    """Convert state-value grids into rounded lists for JSON output."""

    grids = value_grids(q_table)
    return {label: grid.round(6).tolist() for label, grid in grids.items()}


def save_outputs(
    config: Config,
    q_table: np.ndarray,
    visit_counts: np.ndarray,
    rewards: list[float],
    metrics: dict[str, float],
) -> Path:
    """Persist plots and summary metadata for a training run."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_reward_curve(rewards, config.moving_avg_window, output_dir / "reward_curve.png")
    save_policy_plots(q_table, output_dir)
    save_value_plots(q_table, output_dir)

    summary = {
        "config": asdict(config),
        "train": {
            "episodes": config.episodes,
            "avg_reward_last_5000": float(np.mean(rewards[-5000:])) if rewards else 0.0,
            "best_episode_reward": float(np.max(rewards)) if rewards else 0.0,
        },
        "eval": metrics,
        "policy": {
            "row_labels": PLAYER_SUMS,
            "column_labels": DEALER_SHOWING,
            "no_usable_ace": policy_tables(q_table)["no_usable_ace"],
            "usable_ace": policy_tables(q_table)["usable_ace"],
        },
        "state_values": {
            "row_labels": PLAYER_SUMS,
            "column_labels": DEALER_SHOWING,
            "no_usable_ace": value_tables(q_table)["no_usable_ace"],
            "usable_ace": value_tables(q_table)["usable_ace"],
        },
        "visit_counts": visit_counts.tolist(),
        "q_table": q_table.round(6).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_policy_table(title: str, table: list[list[str]]) -> None:
    """Print a compact policy table for the CLI."""

    print(title)
    print("Player\\Dealer  " + "  ".join(f"{card:>2}" for card in DEALER_SHOWING))
    for player_sum, row in zip(PLAYER_SUMS, table, strict=True):
        print(f"{player_sum:>13}  " + "  ".join(f"{action[0]}" for action in row))
    print()


def main() -> None:
    """Run training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    q_table, visit_counts, rewards = train(config)
    metrics = evaluate(config, q_table)
    output_dir = save_outputs(config, q_table, visit_counts, rewards, metrics)

    print(f"Run saved to: {output_dir}")
    print(f"Evaluation avg_reward: {metrics['avg_reward']:.4f}")
    print(f"Evaluation win_rate: {metrics['win_rate']:.4f}")
    print(f"Evaluation draw_rate: {metrics['draw_rate']:.4f}")
    print(f"Evaluation loss_rate: {metrics['loss_rate']:.4f}")
    print(f"Evaluation avg_episode_length: {metrics['avg_episode_length']:.4f}")

    if config.render_final_policy:
        tables = policy_tables(q_table)
        print()
        print_policy_table("Greedy policy without usable ace:", tables["no_usable_ace"])
        print_policy_table("Greedy policy with usable ace:", tables["usable_ace"])


if __name__ == "__main__":
    main()
