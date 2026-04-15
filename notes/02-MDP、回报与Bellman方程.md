# MDP、回报与Bellman方程

本节汇集强化学习中的几个基础概念：问题抽象、回报定义和 Bellman 递推。

## MDP 是什么

很多强化学习问题都可以写成马尔可夫决策过程：

$$
(S, A, P, R, \gamma)
$$

- `S`：状态空间
- `A`：动作空间
- `P`：状态转移概率
- `R`：奖励函数
- $\gamma$：折扣因子

其中最关键的是“马尔可夫性”：下一步只依赖当前状态和动作，而不依赖更早的历史。

## 回报和价值函数

强化学习不只看单步奖励，而是看从当前时刻开始的累计回报：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

在这个基础上，会定义两类常见值函数：

- $V^\pi(s)$：在策略 $\pi$ 下，从状态 $s$ 出发的期望回报
- $Q^\pi(s, a)$：在状态 $s$ 先做动作 $a$，之后继续按策略行动时的期望回报

状态价值函数与动作价值函数分别对应状态起点视角和状态动作起点视角。

## Bellman 方程在说什么

Bellman 方程将当前价值拆分为“当前一步奖励 + 下一步价值的折扣和”。

状态价值函数满足：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

动作价值函数满足：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a \right]
$$

很多算法的区别，最后都体现在“下一步价值到底怎么估计”。

## Q-Learning 作为起点的原因

`Q-Learning` 常作为起点，原因主要是：

- 不需要神经网络
- 直接围绕 $Q(s, a)$ 更新
- 在小型离散环境里，公式与代码的对应关系更直接

典型更新式：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

该更新过程构成理解 `SARSA`、`Monte Carlo` 和 `TD` 方法的基础。

## 对应内容

- [01-强化学习、状态、动作与Q值](./01-强化学习、状态、动作与Q值.md)
- [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md)
- [01-frozenlake-tabular-q](../experiments/01-frozenlake-tabular-q/README.md)
