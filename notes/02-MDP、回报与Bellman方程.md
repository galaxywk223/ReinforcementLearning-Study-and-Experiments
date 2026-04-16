# MDP、回报与Bellman方程

本章将强化学习问题抽象为马尔可夫决策过程，并给出回报定义与 Bellman 递推关系。

## 本章目标

- 建立 `MDP` 五元组的统一问题表达。
- 明确累计回报与状态/动作价值函数的定义。
- 理解 Bellman 方程在值函数更新中的作用。
- 说明 `Q-Learning` 作为后续算法入口的原因。

## 核心概念

### 马尔可夫决策过程

强化学习问题常写为：

$$
(S, A, P, R, \gamma)
$$

- `S`：状态空间
- `A`：动作空间
- `P`：状态转移概率
- `R`：奖励函数
- $\gamma$：折扣因子

关键假设是马尔可夫性：下一步只依赖当前状态与动作。

### 回报与值函数

累计回报定义为：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

常见值函数如下：

- $V^\pi(s)$：策略 $\pi$ 下状态 $s$ 的期望回报
- $Q^\pi(s, a)$：策略 $\pi$ 下状态动作对 $(s,a)$ 的期望回报

## 关键关系与公式

### Bellman 状态价值关系

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

### Bellman 动作价值关系

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a \right]
$$

### `Q-Learning` 更新入口

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

该关系是后续 `SARSA`、`Monte Carlo` 与 `n-step` 方法对比的基线。

## 关联实验

- [01-frozenlake-tabular-q](../experiments/01-frozenlake-tabular-q/README.md)
- [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md)

## 小结

`MDP` 提供问题抽象，回报定义提供优化目标，Bellman 方程提供递推结构。多数强化学习算法都可视为“在不同估计假设下构造 Bellman 目标”。

## 继续阅读

- [01-强化学习、状态、动作与Q值](./01-强化学习、状态、动作与Q值.md)
- [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md)
