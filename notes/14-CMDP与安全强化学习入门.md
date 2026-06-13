# CMDP与安全强化学习入门

本章把普通 `MDP` 扩展到带约束的决策问题：在奖励目标之外显式加入 `cost` 与 `constraint`，用于描述安全、资源、风险或合规限制。

## 本章目标

- 区分 `reward` 与 `cost` 在优化问题中的位置。
- 理解 `CMDP` 如何在 `MDP` 基础上表达约束预算。
- 建立 `cost value` 与安全强化学习方法之间的连接。
- 区分期望安全、轨迹级安全与状态级硬安全。

## 核心机制

### 从 `MDP` 到 `CMDP`

普通 `MDP` 通常写作：

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

`CMDP` 在此基础上增加 cost function 与 cost budget：

$$
(\mathcal{S}, \mathcal{A}, P, R, C, \gamma, d)
$$

其中 $C(s,a,s')$ 表示成本函数，$d$ 表示允许的成本预算。`CMDP` 的核心变化不是增加一种负奖励，而是把优化目标和约束条件分开表达。

### Reward 与 Cost

`Reward` 表示需要最大化的任务收益，`cost` 表示需要控制在预算内的风险或资源消耗。

| 项目 | Reward | Cost |
| --- | --- | --- |
| 作用 | 表示收益或任务目标 | 表示风险、资源消耗或违规程度 |
| 优化方向 | 越高越好 | 不超过预算 |
| 典型例子 | 到达目的地、完成任务、获得分数 | 碰撞风险、能耗、延迟、违规次数 |
| 数学位置 | 目标函数 | 约束条件 |

固定惩罚形式可以把 cost 扣进 reward，但该方式不直接表达预算 $d$，也难以说明最终策略是否满足约束。

### 安全约束层次

安全约束可以按强度分为三类：

| 类型 | 含义 | 特点 |
| --- | --- | --- |
| 期望安全 | $\mathbb{E}[\text{累计 cost}] \le d$ | `CMDP` 中最常见，优化相对直接 |
| 轨迹级安全 | 每条 trajectory 的 cost 都不能超过预算 | 比期望约束更严格 |
| 状态级硬安全 | 每个时刻都必须处于安全状态集合 | 常见于安全关键控制 |

期望安全约束允许不同 episode 的 cost 存在波动，只要求长期平均满足预算。轨迹级安全和状态级硬安全更接近部署阶段的强约束要求。

## 关键关系与公式

### CMDP 目标形式

策略 $\pi$ 下的 reward return 可写为：

$$
J_R(\pi)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\right]
$$

cost return 可写为：

$$
J_C(\pi)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t C_{t+1}\right]
$$

`CMDP` 的典型目标为：

$$
\max_\pi J_R(\pi)
$$

subject to:

$$
J_C(\pi) \le d
$$

该形式表示策略既要提高长期 reward，也要把长期 cost 控制在预算以内。

### Cost Value

`CMDP` 中可以同时维护 reward value 与 cost value：

$$
V_R^\pi(s)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\mid S_0=s\right]
$$

$$
V_C^\pi(s)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t C_{t+1}\mid S_0=s\right]
$$

reward value 衡量任务收益，cost value 衡量约束消耗。后续约束强化学习方法通常需要估计 cost return、cost value 或 cost advantage。

### Safe RL 与 CMDP

`Safe RL` 指安全强化学习，研究策略如何在最大化 reward 的同时避免危险行为或违反安全要求。`CMDP` 是其中常见的数学建模方式，但 `Safe RL` 的范围更宽，还包括安全探索、硬约束、风险敏感、不确定性处理和外部安全机制。

关系可以概括为：

```text
MDP -> CMDP formulation -> Safe RL methods
```

## 小结

`CMDP` 在 `MDP` 的状态、动作、转移、奖励和折扣因子之外加入 cost 与 constraint。Reward 表示任务收益，cost 表示必须受控的风险或资源消耗。`CMDP` 的目标是在 cost 不超过预算的前提下最大化长期 reward。`Safe RL` 以 `CMDP` 为重要基础，但还覆盖训练过程安全、部署安全、硬约束和风险敏感等更宽问题。

## 继续阅读

- [15-约束强化学习与Lagrangian方法](./15-约束强化学习与Lagrangian方法.md)
- [12-PPO的裁剪目标与稳定策略更新](./12-PPO的裁剪目标与稳定策略更新.md)
