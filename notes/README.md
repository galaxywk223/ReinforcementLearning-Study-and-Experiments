# 学习笔记

`notes/` 构成仓库的主阅读层。根目录 [README](../README.md) 提供整体导航，[experiments/README.md](../experiments/README.md) 提供实验索引与运行入口。

## 推荐阅读顺序

| 章节 | 主题 | 章节作用 | 实验入口 |
| --- | --- | --- | --- |
| [00-环境安装与运行](./00-环境安装与运行.md) | 环境与命令 | 统一环境准备与运行入口 | [实验索引](../experiments/README.md) |
| [01-强化学习、状态、动作与Q值](./01-强化学习、状态、动作与Q值.md) | 强化学习基础概念 | 状态、动作与动作价值函数的最小定义 | - |
| [02-MDP、回报与Bellman方程](./02-MDP、回报与Bellman方程.md) | MDP 与 Bellman 方程 | 价值递推关系与算法入口公式 | - |
| [03-动态规划的策略评估、策略迭代与价值迭代](./03-动态规划的策略评估、策略迭代与价值迭代.md) | 动态规划 | 已知模型下的 Bellman 求解器 | [FrozenLake DP 实验](../experiments/01-frozenlake-dp/README.md) |
| [04-Q-Learning的值传播与Q表更新](./04-Q-Learning的值传播与Q表更新.md) | Q-Learning | 奖励沿成功轨迹逐轮向前传播 | [FrozenLake 实验](../experiments/02-frozenlake-tabular-q/README.md) |
| [05-SARSA的时序更新与策略差异](./05-SARSA的时序更新与策略差异.md) | SARSA | on-policy 目标如何影响路径风险偏好 | [CliffWalking 实验](../experiments/03-cliffwalking-tabular-sarsa/README.md) |
| [06-MonteCarlo的整局回报与动作价值更新](./06-MonteCarlo的整局回报与动作价值更新.md) | Monte Carlo Control | 整局回报统计如何塑造策略边界 | [Blackjack 实验](../experiments/04-blackjack-monte-carlo/README.md) |
| [07-n-step-SARSA的多步回报与折中更新](./07-n-step-SARSA的多步回报与折中更新.md) | n-step SARSA | 多步真实奖励与自举项的折中更新 | [CliffWalking n-step 实验](../experiments/05-cliffwalking-n-step-sarsa/README.md) |
| [08-Dyna-Q的模型学习与规划更新](./08-Dyna-Q的模型学习与规划更新.md) | Dyna-Q | 模型学习与规划更新如何配合 | [CliffWalking Dyna-Q 实验](../experiments/06-cliffwalking-dyna-q/README.md) |
| [09-DQN的经验回放与目标网络](./09-DQN的经验回放与目标网络.md) | DQN | 从表格值函数过渡到深度值函数 | [CartPole DQN 实验](../experiments/07-cartpole-dqn/README.md) |
| [10-REINFORCE的回合策略梯度与高方差问题](./10-REINFORCE的回合策略梯度与高方差问题.md) | REINFORCE | 直接策略梯度与高方差来源 | [CartPole REINFORCE 实验](../experiments/08-cartpole-reinforce/README.md) |
| [11-Actor-Critic的价值基线与同步更新](./11-Actor-Critic的价值基线与同步更新.md) | Actor-Critic | 用价值基线稳定策略梯度 | [CartPole Actor-Critic 实验](../experiments/09-cartpole-actor-critic/README.md) |
| [12-PPO的裁剪目标与稳定策略更新](./12-PPO的裁剪目标与稳定策略更新.md) | PPO | 裁剪目标与批量 on-policy 更新 | [LunarLander PPO 实验](../experiments/10-lunarlander-ppo/README.md) |
| [13-SAC的最大熵目标与连续动作控制](./13-SAC的最大熵目标与连续动作控制.md) | SAC | 连续动作 off-policy 深度强化学习 | [Pendulum SAC 实验](../experiments/11-pendulum-sac/README.md) |

## 章节模板

- `00` 使用环境模板：`环境准备`、`运行命令`、`运行说明`、`常见问题`、`后续阅读`。
- `01-02` 使用概念模板：`本章目标`、`核心概念`、`关键关系与公式`、`关联实验`、`小结`、`继续阅读`。
- `03-13` 使用算法模板：`本章目标`、`本章实验`、`关键结果`、`核心机制`、`代码与脚本`、`小结`、`继续阅读`。

## 与实验对应

| 实验目录 | 对应章节 | 说明 |
| --- | --- | --- |
| [experiments/01-frozenlake-dp](../experiments/01-frozenlake-dp/README.md) | `03` | 用 `FrozenLake-v1` 观察策略迭代与价值迭代的 Bellman 收敛过程 |
| [experiments/02-frozenlake-tabular-q](../experiments/02-frozenlake-tabular-q/README.md) | `04` | 用 `FrozenLake-v1` 观察 Q 值传播和奖励曲线抬升 |
| [experiments/03-cliffwalking-tabular-sarsa](../experiments/03-cliffwalking-tabular-sarsa/README.md) | `05` | 用 `CliffWalking-v1` 对比风险敏感的 on-policy 更新 |
| [experiments/04-blackjack-monte-carlo](../experiments/04-blackjack-monte-carlo/README.md) | `06` | 用 `Blackjack-v1` 观察策略边界如何逐渐成形 |
| [experiments/05-cliffwalking-n-step-sarsa](../experiments/05-cliffwalking-n-step-sarsa/README.md) | `07` | 用 `CliffWalking-v1` 观察多步回报如何改变信用分配节奏 |
| [experiments/06-cliffwalking-dyna-q](../experiments/06-cliffwalking-dyna-q/README.md) | `08` | 用 `CliffWalking-v1` 观察模型学习与规划更新如何提升样本利用率 |
| [experiments/07-cartpole-dqn](../experiments/07-cartpole-dqn/README.md) | `09` | 用 `CartPole-v1` 观察经验回放与目标网络如何稳定深度值函数更新 |
| [experiments/08-cartpole-reinforce](../experiments/08-cartpole-reinforce/README.md) | `10` | 用 `CartPole-v1` 观察整局回报如何直接驱动策略梯度 |
| [experiments/09-cartpole-actor-critic](../experiments/09-cartpole-actor-critic/README.md) | `11` | 用 `CartPole-v1` 观察价值基线如何稳定策略梯度 |
| [experiments/10-lunarlander-ppo](../experiments/10-lunarlander-ppo/README.md) | `12` | 用 `LunarLander-v3` 观察裁剪目标与 `GAE` 的稳定化作用 |
| [experiments/11-pendulum-sac](../experiments/11-pendulum-sac/README.md) | `13` | 用 `Pendulum-v1` 观察最大熵目标如何扩展到连续动作控制 |
