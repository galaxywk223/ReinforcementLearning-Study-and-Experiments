"""Train a soft actor-critic agent on Pendulum-v1."""

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
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


@dataclass
class Config:
    """Configuration for a single Pendulum SAC run."""

    total_env_steps: int = 10000
    batch_size: int = 128
    buffer_capacity: int = 100000
    learning_starts: int = 1000
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    eval_episodes: int = 20
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "pendulum-sac-baseline"
    print_eval_rollout: bool = False


@dataclass
class Transition:
    """A single replay-buffer transition."""

    state: np.ndarray
    action: np.ndarray
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


class GaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing."""

    def __init__(self, state_dim: int, action_space: gym.spaces.Box, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, int(np.prod(action_space.shape)))
        self.log_std_layer = nn.Linear(hidden_dim, int(np.prod(action_space.shape)))
        action_scale = (action_space.high - action_space.low) / 2.0
        action_bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), min=-5.0, max=2.0)
        return mean, log_std

    def sample(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(inputs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        raw_action = distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(raw_action)
        correction = torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=1)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """State-action value network used by SAC critics."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([states, actions], dim=1)).squeeze(-1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a soft actor-critic agent on Pendulum-v1 and save the resulting metrics."
    )
    parser.add_argument("--total-env-steps", type=int, default=10000, help="Total environment steps collected.")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size used for replay updates.")
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=100000,
        help="Maximum number of transitions kept in the replay buffer.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Number of environment steps collected before the first gradient update.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by all MLPs.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for soft Q targets.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft-update coefficient for target critics.")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="Learning rate for the policy network.")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="Learning rate for both critic networks.")
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=3e-4,
        help="Learning rate for the automatic entropy coefficient.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes.")
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
        default="pendulum-sac-baseline",
        help="Name of the output subdirectory.",
    )
    parser.add_argument(
        "--print-eval-rollout",
        action="store_true",
        help="Print a sample deterministic evaluation rollout after training.",
    )
    args = parser.parse_args()
    return Config(
        total_env_steps=args.total_env_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        print_eval_rollout=args.print_eval_rollout,
    )


def make_env(seed: int) -> gym.Env:
    """Create a seeded Pendulum environment."""

    env = gym.make("Pendulum-v1")
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


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Softly blend source parameters into the target network."""

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def train(config: Config) -> tuple[GaussianPolicy, list[float], list[float], dict[str, float]]:
    """Train SAC and return reward and critic-loss histories."""

    set_global_seeds(config.seed)
    device = select_device()
    env = make_env(config.seed)
    rng = np.random.default_rng(config.seed)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    policy = GaussianPolicy(state_dim, env.action_space, config.hidden_dim).to(device)
    q1 = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    q2 = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    target_q1 = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    target_q2 = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    q1_optimizer = torch.optim.Adam(q1.parameters(), lr=config.q_lr)
    q2_optimizer = torch.optim.Adam(q2.parameters(), lr=config.q_lr)
    log_alpha = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
    target_entropy = -float(action_dim)

    replay_buffer = ReplayBuffer(config.buffer_capacity)
    episode_rewards: list[float] = []
    q_losses: list[float] = []
    alpha_values: list[float] = []

    state, _ = env.reset(seed=config.seed)
    running_reward = 0.0
    episode_index = 0

    for step in range(1, config.total_env_steps + 1):
        if step <= config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_tensor, _, _ = policy.sample(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.add(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=done,
            )
        )

        running_reward += reward
        state = next_state

        if len(replay_buffer) >= config.batch_size and step > config.learning_starts:
            transitions = replay_buffer.sample(config.batch_size, rng)
            states = torch.as_tensor(np.stack([item.state for item in transitions]), dtype=torch.float32, device=device)
            actions = torch.as_tensor(
                np.stack([item.action for item in transitions]),
                dtype=torch.float32,
                device=device,
            )
            rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
            next_states = torch.as_tensor(
                np.stack([item.next_state for item in transitions]),
                dtype=torch.float32,
                device=device,
            )
            dones = torch.as_tensor([item.done for item in transitions], dtype=torch.float32, device=device)

            with torch.no_grad():
                next_actions, next_log_probs, _ = policy.sample(next_states)
                target_q = torch.min(target_q1(next_states, next_actions), target_q2(next_states, next_actions))
                alpha = log_alpha.exp()
                td_target = rewards + config.gamma * (1.0 - dones) * (target_q - alpha * next_log_probs)

            current_q1 = q1(states, actions)
            current_q2 = q2(states, actions)
            q1_loss = F.mse_loss(current_q1, td_target)
            q2_loss = F.mse_loss(current_q2, td_target)

            q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_optimizer.step()

            q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_optimizer.step()

            new_actions, log_probs, _ = policy.sample(states)
            q_new_actions = torch.min(q1(states, new_actions), q2(states, new_actions))
            alpha = log_alpha.exp()
            policy_loss = (alpha * log_probs - q_new_actions).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            soft_update(q1, target_q1, config.tau)
            soft_update(q2, target_q2, config.tau)

            q_losses.append(float(0.5 * (q1_loss.item() + q2_loss.item())))
            alpha_values.append(float(log_alpha.exp().item()))

        if done:
            episode_rewards.append(float(running_reward))
            episode_index += 1
            state, _ = env.reset(seed=config.seed + episode_index)
            running_reward = 0.0

    env.close()
    train_stats = {
        "avg_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        "best_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "avg_q_loss_last_50": float(np.mean(q_losses[-50:])) if q_losses else 0.0,
        "final_alpha": float(alpha_values[-1]) if alpha_values else 1.0,
        "replay_buffer_final_size": float(len(replay_buffer)),
        "total_env_steps": float(config.total_env_steps),
        "device": str(device),
    }
    return policy, episode_rewards, q_losses, train_stats


def evaluate(config: Config, policy: GaussianPolicy) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate the deterministic policy and record one representative rollout."""

    device = select_device()
    env = make_env(config.seed + 1)
    rewards: list[float] = []
    episode_lengths: list[int] = []
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
                _, _, mean_action = policy.sample(state_tensor)
                action = mean_action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)

            if sample_rollout is None and len(transitions) < 25:
                transitions.append(
                    {
                        "step": steps + 1,
                        "state": np.asarray(state, dtype=np.float32).round(6).tolist(),
                        "action": np.asarray(action, dtype=np.float32).round(6).tolist(),
                        "reward": float(reward),
                        "next_state": np.asarray(next_state, dtype=np.float32).round(6).tolist(),
                    }
                )

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(float(total_reward))
        episode_lengths.append(steps)

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


def save_curve(values: list[float], window: int, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    """Save a smoothed curve image."""

    smoothed = moving_average(values, window)
    plt.figure(figsize=(10, 4.5))
    plt.plot(smoothed, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    config: Config,
    policy: GaussianPolicy,
    episode_rewards: list[float],
    q_losses: list[float],
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
        title="Pendulum SAC Reward Curve",
        xlabel="Episode",
        ylabel=f"Moving average reward (window={min(config.moving_avg_window, len(episode_rewards))})",
        output_path=output_dir / "reward_curve.png",
    )
    save_curve(
        values=q_losses,
        window=max(1, min(config.moving_avg_window, len(q_losses))),
        title="Pendulum SAC Critic Loss Curve",
        xlabel="Optimizer step",
        ylabel=f"Moving average Q loss (window={min(config.moving_avg_window, len(q_losses))})",
        output_path=output_dir / "q_loss_curve.png",
    )

    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": eval_metrics,
        "sample_eval_rollout": sample_rollout,
        "model": {
            "architecture": f"PolicyMLP(3 -> {config.hidden_dim} -> {config.hidden_dim}) + twin Q MLPs",
            "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def print_rollout(rollout: dict[str, object]) -> None:
    """Print a compact version of a sampled evaluation rollout."""

    print(
        "Sample deterministic rollout: "
        f"steps={rollout['steps']}, total_reward={rollout['total_reward']:.1f}, "
        f"terminated={rollout['terminated']}, truncated={rollout['truncated']}"
    )
    for transition in rollout["transitions"]:
        print(
            f"  Step {transition['step']}: "
            f"state={transition['state']} -> action={transition['action']} -> "
            f"reward={transition['reward']:.3f} -> next_state={transition['next_state']}"
        )


def main() -> None:
    """Run SAC training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    policy, episode_rewards, q_losses, train_stats = train(config)
    eval_metrics, sample_rollout = evaluate(config, policy)
    output_dir = save_outputs(
        config=config,
        policy=policy,
        episode_rewards=episode_rewards,
        q_losses=q_losses,
        train_stats=train_stats,
        eval_metrics=eval_metrics,
        sample_rollout=sample_rollout,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_20: {train_stats['avg_reward_last_20']:.4f}")
    print(f"Training avg_q_loss_last_50: {train_stats['avg_q_loss_last_50']:.6f}")
    print(f"Evaluation avg_reward: {eval_metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_episode_length: {eval_metrics['avg_episode_length']:.4f}")

    if config.print_eval_rollout:
        print_rollout(sample_rollout)


if __name__ == "__main__":
    main()
