"""Trace how n-step SARSA updates are delayed and propagated on a fixed CliffWalking path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from train import ACTION_SYMBOLS, make_env


SCRIPTED_ACTIONS = [0] + [1] * 11 + [2]


@dataclass
class TraceConfig:
    """Configuration for the fixed-path n-step SARSA trace."""

    episodes: int = 3
    alpha: float = 0.5
    gamma: float = 0.99
    n_step: int = 4
    seed: int = 42


def parse_args() -> TraceConfig:
    parser = argparse.ArgumentParser(description="Trace how tabular n-step SARSA updates propagate on CliffWalking.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of scripted successful episodes to replay.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate used in the trace.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in the trace.")
    parser.add_argument("--n-step", type=int, default=4, help="Number of rewards accumulated before each update.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment reset.")
    args = parser.parse_args()
    if args.n_step < 1:
        parser.error("--n-step must be at least 1.")
    return TraceConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        n_step=args.n_step,
        seed=args.seed,
    )


def state_to_row_col(state: int, width: int = 12) -> tuple[int, int]:
    """Convert a flattened state index into row and column coordinates."""

    return divmod(state, width)


def format_state(state: int) -> str:
    """Format a state index as 'index(row,col)'."""

    row, col = state_to_row_col(state)
    return f"{state}({row},{col})"


def format_pair(state: int, action: int) -> str:
    """Format a state-action pair for trace output."""

    return f"Q({format_state(state)}, {ACTION_SYMBOLS[action]})"


def print_header(config: TraceConfig) -> None:
    """Print the environment layout and scripted action sequence."""

    print(f"CliffWalking fixed path {config.n_step}-step SARSA trace")
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


def discounted_return(
    q_table: np.ndarray,
    states: list[int],
    rewards: list[float],
    tau: int,
    terminal_time: int,
    config: TraceConfig,
) -> tuple[float, str]:
    """Compute the n-step target for a scripted trace update."""

    upper_bound = min(tau + config.n_step, terminal_time)
    target = 0.0
    for index in range(tau + 1, upper_bound + 1):
        target += (config.gamma ** (index - tau - 1)) * rewards[index]

    if tau + config.n_step < terminal_time:
        bootstrapped_state = states[tau + config.n_step]
        bootstrapped_action = SCRIPTED_ACTIONS[tau + config.n_step]
        bootstrap_value = float(q_table[bootstrapped_state, bootstrapped_action])
        target += (config.gamma**config.n_step) * bootstrap_value
        bootstrap_label = f"{format_pair(bootstrapped_state, bootstrapped_action)}={bootstrap_value:.6f}"
    else:
        bootstrap_label = "none (episode ended before bootstrap term)"

    return target, bootstrap_label


def trace(config: TraceConfig) -> None:
    """Replay a fixed safe path and show each delayed n-step update."""

    env = make_env(config.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

    print_header(config)
    print("Initial Q-table values are all zero.")
    print_path_values(q_table)

    for episode in range(1, config.episodes + 1):
        states: list[int] = []
        actions: list[int] = []
        rewards = [0.0]
        state, _ = env.reset()
        states.append(state)
        print(f"Episode {episode}")

        for time_step, action in enumerate(SCRIPTED_ACTIONS):
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            actions.append(action)
            states.append(next_state)

            print(
                f"  Time {time_step + 1:>2}: "
                f"{format_state(state)} --{ACTION_SYMBOLS[action]}--> {format_state(next_state)} | reward={reward:.1f}"
            )

            tau = time_step - config.n_step + 1
            if tau < 0:
                remaining = -tau
                print(f"           No update yet. Need {remaining} more step(s) before the oldest pair can be updated.")
            else:
                target, bootstrap_label = discounted_return(
                    q_table=q_table,
                    states=states,
                    rewards=rewards,
                    tau=tau,
                    terminal_time=len(SCRIPTED_ACTIONS),
                    config=config,
                )
                update_state = states[tau]
                update_action = actions[tau]
                old_q = float(q_table[update_state, update_action])
                new_q = old_q + config.alpha * (target - old_q)
                q_table[update_state, update_action] = new_q
                print(
                    f"           Update tau={tau}: {format_pair(update_state, update_action)} | "
                    f"return={target:.6f}, bootstrap={bootstrap_label}, old={old_q:.6f}, new={new_q:.6f}"
                )

            state = next_state
            if terminated or truncated:
                break

        terminal_time = len(actions)
        for tau in range(max(0, terminal_time - config.n_step + 1), terminal_time):
            target, bootstrap_label = discounted_return(
                q_table=q_table,
                states=states,
                rewards=rewards,
                tau=tau,
                terminal_time=terminal_time,
                config=config,
            )
            update_state = states[tau]
            update_action = actions[tau]
            old_q = float(q_table[update_state, update_action])
            new_q = old_q + config.alpha * (target - old_q)
            q_table[update_state, update_action] = new_q
            print(
                f"  Tail  tau={tau}: {format_pair(update_state, update_action)} | "
                f"return={target:.6f}, bootstrap={bootstrap_label}, old={old_q:.6f}, new={new_q:.6f}"
            )

        print_path_values(q_table)

    env.close()


def main() -> None:
    """Run the CLI entrypoint."""

    config = parse_args()
    trace(config)


if __name__ == "__main__":
    main()
