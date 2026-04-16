"""Train a PPO-Clip agent on LunarLander-v3."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class Config:
    """Configuration for a single LunarLander PPO run."""

    total_env_steps: int = 200000
    num_envs: int = 8
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    eval_episodes: int = 20
    seed: int = 42
    moving_avg_window: int = 20
    run_name: str = "lunarlander-ppo-baseline"
    print_eval_rollout: bool = False


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic network for PPO."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a PPO-Clip agent on LunarLander-v3 and save the resulting metrics."
    )
    parser.add_argument("--total-env-steps", type=int, default=200000, help="Total environment steps collected.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Number of steps collected per PPO rollout.")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of PPO epochs per rollout batch.")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size used by PPO updates.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size used by both MLP layers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda parameter used by GAE.")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO ratio clipping threshold.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value-loss coefficient.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
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
        default="lunarlander-ppo-baseline",
        help="Name of the output subdirectory.",
    )
    parser.add_argument(
        "--print-eval-rollout",
        action="store_true",
        help="Print a sample greedy evaluation rollout after training.",
    )
    args = parser.parse_args()
    return Config(
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        moving_avg_window=args.moving_avg_window,
        run_name=args.run_name,
        print_eval_rollout=args.print_eval_rollout,
    )


def make_env(seed: int):
    """Build one seeded LunarLander environment factory."""

    def thunk() -> gym.Env:
        env = gym.make("LunarLander-v3")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


def make_eval_env(seed: int) -> gym.Env:
    """Create a seeded single-environment instance for evaluation."""

    env = gym.make("LunarLander-v3")
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


def train(config: Config) -> tuple[ActorCriticNetwork, list[float], list[float], dict[str, float]]:
    """Train PPO and return reward and loss histories."""

    set_global_seeds(config.seed)
    device = select_device()
    envs = gym.vector.SyncVectorEnv([make_env(config.seed + index) for index in range(config.num_envs)])
    obs, _ = envs.reset(seed=[config.seed + index for index in range(config.num_envs)])
    model = ActorCriticNetwork(8, envs.single_action_space.n, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    rollout_batch_size = config.num_envs * config.rollout_steps
    num_updates = max(1, config.total_env_steps // rollout_batch_size)
    episode_rewards: list[float] = []
    loss_history: list[float] = []
    clip_fractions: list[float] = []
    running_returns = np.zeros(config.num_envs, dtype=np.float64)

    for _ in range(num_updates):
        observations = torch.zeros((config.rollout_steps, config.num_envs, 8), dtype=torch.float32, device=device)
        actions = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.int64, device=device)
        log_probs = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
        rewards = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
        dones = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
        values = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)

        for step in range(config.rollout_steps):
            observations[step] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(observations[step])
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            dones[step] = torch.as_tensor(done.astype(np.float32), dtype=torch.float32, device=device)

            running_returns += reward
            for env_index, finished in enumerate(done):
                if finished:
                    episode_rewards.append(float(running_returns[env_index]))
                    running_returns[env_index] = 0.0

            obs = next_obs

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, next_value = model(next_obs_tensor)

        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

        for step in reversed(range(config.rollout_steps)):
            if step == config.rollout_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]
            delta = rewards[step] + config.gamma * next_values * next_non_terminal - values[step]
            last_advantage = delta + config.gamma * config.gae_lambda * next_non_terminal * last_advantage
            advantages[step] = last_advantage

        returns = advantages + values
        batch_obs = observations.reshape((-1, 8))
        batch_actions = actions.reshape(-1)
        batch_log_probs = log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(unbiased=False) + 1e-8)
        indices = np.arange(rollout_batch_size)
        minibatch_losses: list[float] = []
        minibatch_clips: list[float] = []

        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_batch_size, config.minibatch_size):
                minibatch_indices = indices[start : start + config.minibatch_size]
                minibatch_tensor = torch.as_tensor(minibatch_indices, dtype=torch.int64, device=device)

                logits, new_values = model(batch_obs[minibatch_tensor])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(batch_actions[minibatch_tensor])
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_log_probs[minibatch_tensor])
                unclipped_objective = ratio * batch_advantages[minibatch_tensor]
                clipped_objective = torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                ) * batch_advantages[minibatch_tensor]
                policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
                value_loss = 0.5 * (batch_returns[minibatch_tensor] - new_values).pow(2).mean()
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                minibatch_losses.append(float(loss.item()))
                minibatch_clips.append(float(((ratio - 1.0).abs() > config.clip_coef).float().mean().item()))

        loss_history.append(float(np.mean(minibatch_losses)))
        clip_fractions.append(float(np.mean(minibatch_clips)))

    envs.close()
    train_stats = {
        "avg_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
        "best_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "avg_loss_last_10": float(np.mean(loss_history[-10:])) if loss_history else 0.0,
        "avg_clip_fraction_last_10": float(np.mean(clip_fractions[-10:])) if clip_fractions else 0.0,
        "num_updates": float(num_updates),
        "total_env_steps": float(num_updates * rollout_batch_size),
        "device": str(device),
    }
    return model, episode_rewards, loss_history, train_stats


def evaluate(config: Config, model: ActorCriticNetwork) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate the greedy policy and record one representative rollout."""

    device = select_device()
    env = make_eval_env(config.seed + 1)
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
                logits, _ = model(state_tensor)
                distribution = Categorical(logits=logits)
                action = int(distribution.sample().item())
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
        if total_reward >= 200.0:
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
    model: ActorCriticNetwork,
    episode_rewards: list[float],
    loss_history: list[float],
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
        title="LunarLander PPO Reward Curve",
        xlabel="Episode",
        ylabel=f"Moving average reward (window={min(config.moving_avg_window, len(episode_rewards))})",
        output_path=output_dir / "reward_curve.png",
    )
    save_curve(
        values=loss_history,
        window=max(1, min(5, len(loss_history))),
        title="LunarLander PPO Loss Curve",
        xlabel="PPO update",
        ylabel=f"Moving average loss (window={min(5, len(loss_history))})",
        output_path=output_dir / "loss_curve.png",
    )

    summary = {
        "config": asdict(config),
        "train": train_stats,
        "eval": eval_metrics,
        "sample_eval_rollout": sample_rollout,
        "model": {
            "architecture": f"SharedMLP(8 -> {config.hidden_dim} -> {config.hidden_dim}) + policy/value heads",
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
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
    """Run PPO training, evaluation, and artifact generation from the CLI."""

    config = parse_args()
    model, episode_rewards, loss_history, train_stats = train(config)
    eval_metrics, sample_rollout = evaluate(config, model)
    output_dir = save_outputs(
        config=config,
        model=model,
        episode_rewards=episode_rewards,
        loss_history=loss_history,
        train_stats=train_stats,
        eval_metrics=eval_metrics,
        sample_rollout=sample_rollout,
    )

    print(f"Run saved to: {output_dir}")
    print(f"Training avg_reward_last_20: {train_stats['avg_reward_last_20']:.4f}")
    print(f"Training avg_loss_last_10: {train_stats['avg_loss_last_10']:.6f}")
    print(f"Evaluation avg_reward: {eval_metrics['avg_reward']:.4f}")
    print(f"Evaluation avg_episode_length: {eval_metrics['avg_episode_length']:.4f}")
    print(f"Evaluation success_rate: {eval_metrics['success_rate']:.4f}")

    if config.print_eval_rollout:
        print_rollout(sample_rollout)


if __name__ == "__main__":
    main()
