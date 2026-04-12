# 实验代码

这个目录收录仓库中的可运行实验。每个实验都有独立的 `README`，笔记负责解释概念，实验负责把对应方法跑起来。

## 当前实验

| 目录 | 环境 | 方法 | 说明 |
| --- | --- | --- | --- |
| [01-frozenlake-tabular-q](./01-frozenlake-tabular-q/README.md) | `FrozenLake-v1` | `Tabular Q-Learning` | 观察 Q 值如何沿成功路径逐步传播 |
| [02-cliffwalking-tabular-sarsa](./02-cliffwalking-tabular-sarsa/README.md) | `CliffWalking-v1` | `Tabular SARSA` | 观察 `on-policy` 更新与风险规避的关系 |
| [03-blackjack-monte-carlo](./03-blackjack-monte-carlo/README.md) | `Blackjack-v1` | `First-Visit Monte Carlo Control` | 观察整局回报如何更新动作价值 |

## 运行前准备

推荐在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

或者：

```bash
pip install -r requirements.txt
```

## 对应笔记

- [01-第一次理解强化学习](../notes/01-第一次理解强化学习.md)
- [02-MDP、回报与Bellman方程](../notes/02-MDP、回报与Bellman方程.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](../notes/03-Q-Learning是怎么一步步把Q表学出来的.md)
- [04-SARSA是怎么用下一步真实动作更新Q表的](../notes/04-SARSA是怎么用下一步真实动作更新Q表的.md)
- [05-SARSA和Q-Learning在CliffWalking里会学出什么区别](../notes/05-SARSA和Q-Learning在CliffWalking里会学出什么区别.md)
- [06-MonteCarlo是怎么用整局回报更新动作价值的](../notes/06-MonteCarlo是怎么用整局回报更新动作价值的.md)
