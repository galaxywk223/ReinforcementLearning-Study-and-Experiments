# 学习笔记

`notes/` 是这个仓库的主阅读层。建议先看根目录 [README](../README.md)，再按下面的顺序继续阅读；如果只想运行代码，可以直接去 [experiments/README.md](../experiments/README.md)。

## 推荐阅读顺序

| 章节 | 主题 | 适合什么时候读 | 实验入口 |
| --- | --- | --- | --- |
| [00-环境安装与运行](./00-环境安装与运行.md) | 环境与命令 | 想先把依赖、目录和运行方式跑通时 | [实验索引](../experiments/README.md) |
| [01-强化学习、状态、动作与Q值](./01-强化学习、状态、动作与Q值.md) | 强化学习基础概念 | 想先建立状态、动作和 Q 值直觉时 | - |
| [02-MDP、回报与Bellman方程](./02-MDP、回报与Bellman方程.md) | MDP 与 Bellman 方程 | 想理解值函数递推和回报定义时 | - |
| [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md) | Q-Learning | 想看奖励如何沿成功轨迹向前传播时 | [FrozenLake 实验](../experiments/01-frozenlake-tabular-q/README.md) |
| [04-SARSA的时序更新与策略差异](./04-SARSA的时序更新与策略差异.md) | SARSA | 想理解 on-policy 更新为什么更保守时 | [CliffWalking 实验](../experiments/02-cliffwalking-tabular-sarsa/README.md) |
| [05-MonteCarlo的整局回报与动作价值更新](./05-MonteCarlo的整局回报与动作价值更新.md) | Monte Carlo Control | 想把整局回报与最终策略边界对应起来时 | [Blackjack 实验](../experiments/03-blackjack-monte-carlo/README.md) |

## 与实验对应

| 实验目录 | 对应章节 | 说明 |
| --- | --- | --- |
| [experiments/01-frozenlake-tabular-q](../experiments/01-frozenlake-tabular-q/README.md) | `03` | 用 `FrozenLake-v1` 观察 Q 值传播和奖励曲线抬升 |
| [experiments/02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md) | `04` | 用 `CliffWalking-v1` 对比风险敏感的 on-policy 更新 |
| [experiments/03-blackjack-monte-carlo](../experiments/03-blackjack-monte-carlo/README.md) | `05` | 用 `Blackjack-v1` 观察策略边界如何逐渐成形 |
