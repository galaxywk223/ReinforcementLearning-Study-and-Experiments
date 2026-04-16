"""Train a vanilla DQN agent on CartPole-v1."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    """Configuration for a single CartPole DQN run."""

    episodes: int = 400
    batch_size: int = 64
    buffer_capacity: int = 20000
    learning_starts: int = 1000
    target_sync_steps: int = 200
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    eval_episodes: int = 50
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "cartpole-dqn-baseline"
    print_eval_rollout: bool = False


@dataclass
class Transition:
    """A single replay-buffer transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[int(index)] for index in indices]


class QNetwork(nn.Module):
    """Two-layer MLP used for DQN."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a vanilla DQN agent on CartPole-v1 and save the resulting metrics."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size used for replay updates.")
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=20000,
        help="Maximum number of transitions kept in the replay buffer.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Number of environment steps collected before the first gradient update.",
    )
    parser.add_argument(
        "--target-sync-steps",
        type=int,
        default=200,
        help="Number of optimizer steps between target-network hard updates.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by both MLP layers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for TD targets.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=0.99, help="Per-episode epsilon decay factor.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training and evaluation.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=20,
        help="Window size used when smoothing training curves.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="cartpole-dqn-baseline",
        help="Name of the output subdirectory.",
    )
    parser.add_argument(
        "--print-eval-rollout",
        action="store_true",
        help="Print a sample greedy evaluation rollout after training.",
    )
    args = parser.parse_args()
    return Config(
        episodes=args.episodes,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
        target_sync_steps=args.target_sync_steps,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        print_eval_rollout=args.print_eval_rollout,
    )


def make_env(seed: int) -> gym.Env:
    """Create a seeded CartPole environment."""

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def set_global_seeds(seed: int) -> None:
    """Set random seeds for NumPy and PyTorch."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Prefer CUDA when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy_action(
    policy_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    action_space: gym.Space,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    """Choose an action using epsilon-greedy exploration."""

    if rng.random() < epsilon:
        return int(action_space.sample())

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_values = policy_net(state_tensor)
    return int(torch.argmax(action_values, dim=1).item())


def optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """Run one DQN minibatch update and return the loss value."""

    transitions = replay_buffer.sample(batch_size, rng)
    states = torch.as_tensor(np.stack([item.state for item in transitions]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([item.action for item in transitions], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor([item.done for item in transitions], dtype=torch.float32, device=device)

    predicted_q = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        td_target = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(predicted_q, td_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(config: Config) -> tuple[QNetwork, list[float], list[float], dict[str, float]]:
    """Train the DQN agent and return the learned network and training histories."""

    set_global_seeds(config.seed)
    device = select_device()
    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_capacity)

    episode_rewards: list[float] = []
    losses: list[float] = []
    epsilon = config.epsilon_start
    env_steps = 0
    optimizer_steps = 0

    for _ in range(config.episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = epsilon_greedy_action(policy_net, state, epsilon, env.action_space, rng, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay_buffer.add(
                Transition(
                    state=np.asarray(state, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    next_state=np.asarray(next_state, dtype=np.float32),
                    done=done,
                )
            )

            state = next_state
            total_reward += reward
            env_steps += 1

            if env_steps >= config.learning_starts and len(replay_buffer) >= config.batch_size:
                loss_value = optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=config.batch_size,
                    gamma=config.gamma,
                    rng=rng,
                    device=device,
                )
                losses.append(loss_value)
                optimizer_steps += 1
                if optimizer_steps % config.target_sync_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(float(total_reward))
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    env.close()
    train_stats = {
        "avg_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        "best_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "avg_loss_last_50": float(np.mean(losses[-50:])) if losses else 0.0,
        "replay_buffer_final_size": len(replay_buffer),
        "total_env_steps": env_steps,
        "optimizer_steps": optimizer_steps,
        "final_epsilon": float(epsilon),
        "device": str(device),
    }
    return policy_net, episode_rewards, losses, train_stats


def evaluate(config: Config, policy_net: QNetwork) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate the greedy policy and record one representative rollout."""

    device = select_device()
    env = make_env(config.seed + 1)
    max_episode_steps = int(env.spec.max_episode_steps) if env.spec is not None else 500
    rewards: list[float] = []
    episode_lengths: list[int] = []
    successes = 0
    sample_rollout: dict[str, object] | None = None

    for episode_index in range(config.eval_episodes):
        state, _ = env.reset(seed=config.seed + 1 + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        transitions: list[dict[str, object]] = []

        while not (terminated or truncated):
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(state_tensor).squeeze(0)
                action = int(torch.argmax(q_values).item())
            next_state, reward, terminated, truncated, _ = env.step(action)

            if sample_rollout is None and len(transitions) < 25:
                transitions.append(
                    {
                        "step": steps + 1,
                        "state": np.asarray(state, dtype=np.float32).round(6).tolist(),
                        "action": action,
                        "reward": float(reward),
                        "next_state": np.asarray(next_state, dtype=np.float32).round(6).tolist(),
                    }
                )

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(float(total_reward))
        episode_lengths.append(steps)
        if steps >= max_episode_steps:
            successes += 1

        if sample_rollout is None:
            sample_rollout = {
                "steps": steps,
                "total_reward": float(total_reward),
                "truncated": bool(truncated),
                "terminated": bool(terminated),
                "transitions": transitions,
            }

    env.close()
    assert sample_rollout is not None
    metrics = {
        "avg_reward": float(np.mean(rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(successes / config.eval_episodes),
    }
    return metrics, sample_rollout


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Compute a simple moving average over a history list."""

    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return data
    window = max(1, min(window, len(data)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode="valid")


def save_curve(
    values: list[float],
    window: int,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a smoothed curve image."""

    smoothed = moving_average(values, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(smoothed, linewidth=2)
    plt.title(title)
    plt.xlabel("Episode" if "Reward" in title else "Optimizer step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: Config,
    policy_net: QNetwork,
    episode_rewards: list[float],
    losses: list[float],
    train_stats: dict[str, float],
    eval_metrics: dict[str, float],
    sample_rollout: dict[str, object],
) -> Path:
    """Persist plots, model metadata, and summary JSON."""

    output_dir = Path(__file__).resolve().parent / "outputs" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_curve(
        values=episode_rewards,
        window=config.moving_avg_window,
        title="CartPole DQN Reward Curve",
        ylabel=f"Moving average reward (window={min(config.moving_avg_window, len(episode_rewards))})",
        output_path=output_dir / "reward_curve.png",
    )
    save_curve(
        values=losses,
        window=config.moving_avg_window,
        title="CartPole DQN Loss Curve",
        ylabel=f"Moving average loss (window={min(config.moving_avg_window, len(losses))})",
        output_path=output_dir / "loss_curve.png",
    )

    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": eval_metrics,
        "sample_eval_rollout": sample_rollout,
        "model": {
            "architecture": f"MLP(4 -> {config.hidden_dim} -> {config.hidden_dim} -> 2)",
            "parameter_count": int(sum(parameter.numel() for parameter in policy_net.parameters())),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_rollout(rollout: dict[str, object]) -> None:
    """Print a compact version of a sampled evaluation rollout."""

    print(
        "Sample greedy rollout: "
        f"steps={rollout['steps']}, total_reward={rollout['total_reward']:.1f}, "
        f"terminated={rollout['terminated']}, truncated={rollout['truncated']}"
    )
    for transition in rollout["transitions"]:
        print(
            f"  Step {transition['step']}: "
            f"state={transition['state']} -> action={transition['action']} -> "
            f"reward={transition['reward']:.1f} -> next_state={transition['next_state']}"
        )


def main() -> None:
    """Run training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    policy_net, episode_rewards, losses, train_stats = train(config)
    eval_metrics, sample_rollout = evaluate(config, policy_net)
    output_dir = save_outputs(
        config=config,
        policy_net=policy_net,
        episode_rewards=episode_rewards,
        losses=losses,
        train_stats=train_stats,
        eval_metrics=eval_metrics,
        sample_rollout=sample_rollout,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_20: {train_stats['avg_reward_last_20']:.4f}")
    print(f"Training avg_loss_last_50: {train_stats['avg_loss_last_50']:.6f}")
    print(f"Evaluation avg_reward: {eval_metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_episode_length: {eval_metrics['avg_episode_length']:.4f}")
    print(f"Evaluation success_rate: {eval_metrics['success_rate']:.4f}")

    if config.print_eval_rollout:
        print_rollout(sample_rollout)


if __name__ == "__main__":
    main()
