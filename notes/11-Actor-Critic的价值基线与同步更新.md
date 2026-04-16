# Actor-Critic的价值基线与同步更新

本章给出策略梯度主线的第一种稳定化形式：策略网络负责选动作，价值网络负责提供基线，两者在同一条在线轨迹上同步更新。

## 本章目标

- 说明 `Actor-Critic` 中策略头与价值头的分工。
- 理解一步 `TD error` 如何作为优势近似。
- 观察引入价值基线后，相比 `REINFORCE` 的训练节奏变化。

## 本章实验

- 主环境：`CartPole-v1`
- 方法：共享干路的同步 `Actor-Critic`
- 关键配置：`Shared MLP + policy/value heads`、一步 `TD error`

## 关键结果

当前仓库基线结果如下：

- 回合数：`400`
- 评估平均回报：`500.0`
- 评估平均回合长度：`500.0`
- 成功率：`1.0`
- 对比评估平均回报：`Actor-Critic=500.0`，`REINFORCE=498.0`

<p align="center">
  <img src="../assets/figures/cartpole-actor-critic/comparison_reward_curve.png" alt="CartPole Actor-Critic 与 REINFORCE 对比曲线" width="920" />
</p>

## 核心机制

### Actor 与 Critic

- `Actor`：输出策略 $\pi_\theta(a \mid s)$。
- `Critic`：输出状态价值 $V_w(s)$。

当前实现共享前两层感知机，仅在最后分出策略头和价值头。

### 一步优势近似

当前实现使用一步 `TD error` 近似优势：

$$
\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)
$$

策略头更新时使用 $\delta_t$ 的停止梯度版本作为权重；价值头则直接回归该 `TD target`。

### 同步更新

每执行一步环境交互，就同时完成：

- 一次策略梯度更新
- 一次价值回归更新

因此 `Actor-Critic` 不必像 `REINFORCE` 一样等整局结束后再回传回报。

### 为什么比 `REINFORCE` 更稳

价值基线把“动作本身带来的好坏”与“整局回报整体尺度”区分开，显著降低了策略梯度权重的波动。

## 代码与脚本

### 代码入口

- [train.py](../experiments/09-cartpole-actor-critic/train.py)
- [compare_actor_critic_reinforce.py](../experiments/09-cartpole-actor-critic/compare_actor_critic_reinforce.py)

### 运行命令

```bash
cd experiments/09-cartpole-actor-critic
python train.py --episodes 400
python compare_actor_critic_reinforce.py
```

### 脚本说明

- `train.py`：完整训练、评估与曲线导出。
- `compare_actor_critic_reinforce.py`：对比有无价值基线时的训练曲线。

核心一步优势构造如下：

```python
td_target = reward_tensor + config.gamma * next_value * float(not done)
advantage = td_target - value
actor_loss = -(distribution.log_prob(action) * advantage.detach()).mean()
critic_loss = 0.5 * advantage.pow(2).mean()
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/critic_loss_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 小结

`Actor-Critic` 用价值基线解决了 `REINFORCE` 的主要方差问题，同时保留了直接优化策略的能力。后续 `PPO` 可以看作在这一范式上继续加强稳定性和批量更新能力。

## 继续阅读

- [12-PPO的裁剪目标与稳定策略更新](./12-PPO的裁剪目标与稳定策略更新.md)
- [09-cartpole-actor-critic](../experiments/09-cartpole-actor-critic/README.md)
