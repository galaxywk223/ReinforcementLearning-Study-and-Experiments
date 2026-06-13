# 约束强化学习与Lagrangian方法

本章延续 `CMDP` 的约束建模，说明 `Constrained RL` 如何把 cost constraint 放入策略优化过程，并比较固定惩罚、`Lagrangian`、`PPO-Lagrangian` 与 `CPO` 的差异。

## 本章目标

- 理解 `Constrained RL` 与 `CMDP`、`Safe RL` 的关系。
- 区分固定 `Reward Penalty` 与自适应 `Lagrangian` 方法。
- 理解 primal-dual 更新中策略和对偶变量的分工。
- 说明 `PPO-Lagrangian` 与 `CPO` 的基本思想和约束强度差异。

## 核心机制

### Constrained RL 定位

`CMDP` 提供问题建模，`Constrained RL` 关注求解方法，`Safe RL` 则覆盖更宽的安全强化学习问题。

| 概念 | 侧重点 |
| --- | --- |
| CMDP | 用 `MDP + cost + constraint` 表达问题 |
| Constrained RL | 在 `CMDP` 中学习满足约束的策略 |
| Safe RL | 包含约束、安全探索、硬安全和风险敏感等方向 |

`Constrained RL` 的核心目标仍是提高 reward，但高 reward 策略若 cost 超过预算，则不属于可行策略。

### Reward Penalty 与 Lagrangian

固定惩罚法将 cost 直接扣进 reward：

$$
J_R(\pi)-\alpha J_C(\pi)
$$

其中 $\alpha$ 是人工设置的惩罚系数。该形式简单，但不显式使用预算 $d$，也不保证最终 cost 满足约束。

`Lagrangian` 方法构造带对偶变量的目标：

$$
\mathcal{L}(\pi,\lambda)=J_R(\pi)-\lambda(J_C(\pi)-d)
$$

其中 $\lambda \ge 0$ 是约束对应的对偶变量，可理解为 cost 的影子价格。若 cost 超过预算，$\lambda$ 增大，策略更新会更重视降低 cost；若 cost 低于预算，$\lambda$ 可以减小，策略可以更积极追求 reward。

| 方法 | 惩罚系数 | 是否显式使用预算 $d$ | 约束含义 |
| --- | --- | --- | --- |
| Reward Penalty | 固定 $\alpha$ | 否 | cost 被当作负 reward |
| Lagrangian | 自适应 $\lambda$ | 是 | cost 是需要满足的约束 |

### Primal-Dual 更新

`Lagrangian RL` 通常可以理解为 primal-dual 过程：

- Policy 是 primal variable，负责追求更高 reward。
- $\lambda$ 是 dual variable，负责压住 constraint。

策略更新倾向于最大化：

$$
J_R(\pi)-\lambda J_C(\pi)
$$

$\lambda$ 更新则根据约束违反量调整：

$$
\lambda \leftarrow \left[\lambda+\eta_\lambda(J_C(\pi)-d)\right]_+
$$

其中 $\eta_\lambda$ 是 $\lambda$ 的学习率，$[\cdot]_+$ 表示截断到非负范围。该机制形成反馈：策略越违反约束，cost 惩罚越重；策略越保守且 cost 低于预算，惩罚可以减轻。

## 关键关系与公式

### 固定 Lambda 的限制

固定某个 $\lambda$ 后，`Lagrangian` 目标变成一个无约束优化问题：

$$
\max_\pi \left[J_R(\pi)-\lambda(J_C(\pi)-d)\right]
$$

该问题不等价于原始 `CMDP`。任意固定 $\lambda$ 都可能选出 cost 超标的策略，也可能选出过度保守的策略。`Lagrangian` 方法真正寻找的是 saddle point：

$$
\max_\pi \min_{\lambda \ge 0}\mathcal{L}(\pi,\lambda)
$$

在深度强化学习中，策略通常由神经网络参数化，优化问题非凸且采样带有噪声。因此 `Lagrangian` 更适合作为实用算法框架，而不是无条件的最优性保证。

### PPO-Lagrangian

`PPO-Lagrangian` 将 `Lagrangian` 思想放入 `PPO`。普通 `PPO` 根据 reward advantage 更新策略，`PPO-Lagrangian` 引入 cost advantage 和 $\lambda$，形成混合 advantage：

$$
A_{\text{mix}}=A_R-\lambda A_C
$$

其中 $A_R$ 衡量动作对 reward 的贡献，$A_C$ 衡量动作对 cost 的贡献。实现上通常需要增加 cost critic 与 $\lambda$ 更新。

`PPO-Lagrangian` 的主要风险是 cost oscillation。若 cost 超标，$\lambda$ 增大，策略变保守；若 cost 低于预算，$\lambda$ 减小，策略又可能变激进。该方法更接近“最终或平均意义上控制 cost”，不保证训练过程中每一步都安全。

### CPO

`CPO` 全称为 `Constrained Policy Optimization`。该方法不只是把 cost 变成惩罚，而是在每次策略更新中显式保留约束。其直觉目标是提高 reward，同时避免本次策略更新明显破坏 cost constraint。

`CPO` 通常还加入 KL 约束，限制新策略不要离旧策略太远：

$$
D_{\mathrm{KL}}(\pi_{\text{old}} || \pi_{\text{new}}) \le \delta
$$

相比 `PPO-Lagrangian`，`CPO` 更接近直接求解带约束的策略更新问题，但实现复杂度和计算成本更高。

### 方法对比

| 方法 | 核心思想 | 约束强度 | 实现复杂度 |
| --- | --- | --- | --- |
| Reward Penalty | 固定系数扣 cost | 弱 | 低 |
| PPO-Lagrangian | 自适应 cost penalty | 中等 | 中等 |
| CPO | 显式处理 cost constraint 与 KL constraint | 较强 | 高 |

从约束意识上看：

```text
Reward Penalty < PPO-Lagrangian < CPO
```

## 小结

`Constrained RL` 在强化学习中显式加入约束，目标是在可行策略集合中找到 reward 尽可能高的策略。`Lagrangian` 方法通过对偶变量 $\lambda$ 将约束违反转化为自适应惩罚，但固定 $\lambda$ 不等价于原 `CMDP`。`PPO-Lagrangian` 实现相对直接，但可能出现 cost oscillation。`CPO` 更直接处理带约束的策略更新，约束意识更强，但实现复杂度更高。

## 继续阅读

- [14-CMDP与安全强化学习入门](./14-CMDP与安全强化学习入门.md)
- [12-PPO的裁剪目标与稳定策略更新](./12-PPO的裁剪目标与稳定策略更新.md)
