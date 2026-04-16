# 学习笔记

`notes/` 构成仓库的主阅读层。根目录 [README](../README.md) 提供整体导航，[experiments/README.md](../experiments/README.md) 提供实验索引与运行入口。

## 章节顺序

| 章节 | 主题 | 内容定位 | 实验入口 |
| --- | --- | --- | --- |
| [00-环境安装与运行](./00-环境安装与运行.md) | 环境与命令 | 依赖、目录与运行方式概览 | [实验索引](../experiments/README.md) |
| [01-强化学习、状态、动作与Q值](./01-强化学习、状态、动作与Q值.md) | 强化学习基础概念 | 状态、动作和 Q 值的基础定义 | - |
| [02-MDP、回报与Bellman方程](./02-MDP、回报与Bellman方程.md) | MDP 与 Bellman 方程 | 值函数递推与回报定义 | - |
| [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md) | Q-Learning | 成功轨迹上的值传播过程 | [FrozenLake 实验](../experiments/01-frozenlake-tabular-q/README.md) |
| [04-SARSA的时序更新与策略差异](./04-SARSA的时序更新与策略差异.md) | SARSA | 风险敏感的 on-policy 更新差异 | [CliffWalking 实验](../experiments/02-cliffwalking-tabular-sarsa/README.md) |
| [05-MonteCarlo的整局回报与动作价值更新](./05-MonteCarlo的整局回报与动作价值更新.md) | Monte Carlo Control | 整局回报与策略边界的对应关系 | [Blackjack 实验](../experiments/03-blackjack-monte-carlo/README.md) |
| [06-n-step SARSA的多步回报与折中更新](./06-n-step-SARSA的多步回报与折中更新.md) | n-step SARSA | 多步回报如何折中单步 TD 和整局回报 | [CliffWalking n-step 实验](../experiments/04-cliffwalking-n-step-sarsa/README.md) |
| [07-DQN的经验回放与目标网络](./07-DQN的经验回放与目标网络.md) | DQN | 经验回放与目标网络如何稳定深度值函数学习 | [CartPole DQN 实验](../experiments/05-cartpole-dqn/README.md) |

## 与实验对应

| 实验目录 | 对应章节 | 说明 |
| --- | --- | --- |
| [experiments/01-frozenlake-tabular-q](../experiments/01-frozenlake-tabular-q/README.md) | `03` | 用 `FrozenLake-v1` 观察 Q 值传播和奖励曲线抬升 |
| [experiments/02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md) | `04` | 用 `CliffWalking-v1` 对比风险敏感的 on-policy 更新 |
| [experiments/03-blackjack-monte-carlo](../experiments/03-blackjack-monte-carlo/README.md) | `05` | 用 `Blackjack-v1` 观察策略边界如何逐渐成形 |
| [experiments/04-cliffwalking-n-step-sarsa](../experiments/04-cliffwalking-n-step-sarsa/README.md) | `06` | 用 `CliffWalking-v1` 观察多步回报如何改变信用分配节奏 |
| [experiments/05-cartpole-dqn](../experiments/05-cartpole-dqn/README.md) | `07` | 用 `CartPole-v1` 观察经验回放与目标网络如何稳定深度值函数更新 |
