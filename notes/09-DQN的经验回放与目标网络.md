# DQN的经验回放与目标网络

本章给出从表格型值方法到深度值函数逼近的过渡：在连续状态空间中，用神经网络近似 `Q(s, a)`，并通过经验回放与目标网络稳定训练。

## 本章目标

- 说明表格法在 `CartPole` 连续状态上的局限。
- 理解 `DQN` 的目标值构造与损失定义。
- 明确经验回放和目标网络在稳定性上的作用。

## 本章实验

- 主环境：`CartPole-v1`
- 方法：vanilla `DQN`
- 关键配置：`MLP(4->128->128->2)`、`Huber loss`、经验回放、目标网络硬同步

## 关键结果

当前仓库基线结果如下：

- 回合数：`400`
- 评估平均回报：`500.0`
- 评估平均回合长度：`500.0`
- 成功率：`1.0`

<p align="center">
  <img src="../assets/figures/cartpole-dqn/reward_curve.png" alt="CartPole DQN 奖励曲线" width="920" />
</p>

## 核心机制

### 为什么表格法失效

`CartPole` 状态为连续向量：

$$
s = (x, \dot{x}, \theta, \dot{\theta})
$$

无法穷举维护可用 `Q` 表，需改为参数化函数：

$$
Q_\theta(s, a)
$$

### 目标值与损失

对转移 $(s_t, a_t, r_{t+1}, s_{t+1})$，目标值为：

$$
y_t=
\begin{cases}
r_{t+1}, & \text{终止状态}\\
r_{t+1}+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a'), & \text{非终止状态}
\end{cases}
$$

损失函数：

$$
L(\theta)=\mathbb{E}\left[\ell\left(Q_\theta(s_t,a_t),y_t\right)\right]
$$

当前实现使用 `Huber loss`。

### 经验回放作用

- 将转移写入 `replay buffer`。
- 更新时随机采样 `minibatch`。
- 打散样本相关性，降低梯度抖动与短期覆盖效应。

### 目标网络作用

- `policy_net` 持续更新。
- `target_net` 每隔固定优化步硬同步一次。

当前默认同步间隔为 `200` 优化步。

## 代码与脚本

### 代码入口

- [train.py](../experiments/07-cartpole-dqn/train.py)
- [trace_dqn_updates.py](../experiments/07-cartpole-dqn/trace_dqn_updates.py)

### 运行命令

```bash
cd experiments/07-cartpole-dqn
python train.py --episodes 400 --print-eval-rollout
python trace_dqn_updates.py
```

### 脚本说明

- `train.py`：完整训练、评估与曲线导出，自动设备选择（`cuda` 可用则使用 `cuda`）。
- `trace_dqn_updates.py`：固定批量样本，打印 `Q_pred`、`TD target` 和一次优化前后 `loss`。

核心目标值构造如下：

```python
predicted_q = policy_net(states).gather(1, actions).squeeze(1)
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    td_target = rewards + gamma * next_q * (1.0 - dones)
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/loss_curve.png`

## 小结

`DQN` 的核心改造是“值函数网络化 + 训练稳定化”。经验回放和目标网络并非附加技巧，而是深度值函数可训练性的基础组件。

## 继续阅读

- [10-REINFORCE的回合策略梯度与高方差问题](./10-REINFORCE的回合策略梯度与高方差问题.md)
- [07-cartpole-dqn](../experiments/07-cartpole-dqn/README.md)
