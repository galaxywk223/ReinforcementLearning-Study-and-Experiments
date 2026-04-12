# MDP、回报与Bellman方程

这一篇把强化学习里最常见的几件基础概念放在一起：问题怎么抽象成 MDP，回报怎么看，Bellman 递推在说什么。

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

如果已经习惯了“当前价值”和“未来价值”分开看，这两类定义就比较自然了。

## Bellman 方程在说什么

Bellman 方程的核心意思很简单：当前价值可以拆成“当前一步奖励 + 下一步价值的折扣和”。

状态价值函数满足：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

动作价值函数满足：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a \right]
$$

很多算法的区别，最后都体现在“下一步价值到底怎么估计”。

## 为什么先学 Q-Learning

`Q-Learning` 适合作为起点，原因主要是：

- 不需要神经网络
- 直接围绕 $Q(s, a)$ 更新
- 在小型离散环境里很容易把公式和代码对应起来

典型更新式：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

如果先把这个更新过程看清楚，后面理解 `SARSA`、`Monte Carlo` 和 `TD` 会轻松很多。

## 对应内容

- [01-第一次理解强化学习](./01-第一次理解强化学习.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](./03-Q-Learning是怎么一步步把Q表学出来的.md)
- [01-frozenlake-tabular-q](../experiments/01-frozenlake-tabular-q/README.md)
