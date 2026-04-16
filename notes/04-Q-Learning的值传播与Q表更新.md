# Q-Learning的值传播与Q表更新

本章通过 `FrozenLake` 最小路径追踪说明 `Q-Learning` 的核心现象：价值不会一次写满，而是沿成功轨迹逐轮向前传播。

## 本章目标

- 理解 `Q-Learning` 的目标值构造方式。
- 观察终点奖励如何逐轮传播到更早状态。
- 将公式、追踪脚本输出与完整训练曲线建立对应关系。

## 本章实验

- 主环境：`FrozenLake-v1`
- 追踪设置：`is_slippery=False`，固定成功路径
- 完整训练：默认打滑环境上的表格型 `Q-Learning`

## 关键结果

当前仓库基线结果如下：

- 回合数：`4000`
- 评估平均奖励：`0.73`
- 评估成功率：`0.73`

<p align="center">
  <img src="../assets/figures/frozenlake/reward_curve.png" alt="FrozenLake Q-Learning 奖励曲线" width="920" />
</p>

## 核心机制

### 最小实验设置

固定路径为：

```text
D -> D -> R -> D -> R -> R
```

对应状态序列为：

$$
0 \rightarrow 4 \rightarrow 8 \rightarrow 9 \rightarrow 13 \rightarrow 14 \rightarrow G
$$

仅最后一步 `14 --R--> G` 产生奖励 `1`。

### 更新关系

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

追踪参数为：

$$
\alpha = 0.1,\quad \gamma = 0.99
$$

### 值传播现象

- 第 1 次成功：仅 `Q(14, R)` 先变为正值。
- 第 2 次成功：`Q(13, R)` 开始抬升，价值向前传播一步。
- 第 3 次成功：更早状态（如 `Q(9, D)`）开始抬升。

该顺序由在线更新决定：早期状态只能使用“下一状态当前已有估计”，无法一步拿到完整未来信息。

### 追踪数值样例

| Episode | $Q(0, D)$ | $Q(4, D)$ | $Q(8, R)$ | $Q(9, D)$ | $Q(13, R)$ | $Q(14, R)$ |
| ------- | ---------: | --------: | --------: | --------: | ---------: | ---------: |
| 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 |
| 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.009900 | 0.190000 |
| 3 | 0.000000 | 0.000000 | 0.000000 | 0.000980 | 0.027720 | 0.271000 |

## 代码与脚本

### 代码入口

- [train.py](../experiments/02-frozenlake-tabular-q/train.py)
- [trace_q_updates.py](../experiments/02-frozenlake-tabular-q/trace_q_updates.py)

### 运行命令

```bash
cd experiments/02-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
python trace_q_updates.py --episodes 6
```

### 脚本说明

- `train.py`：完整训练与评估，输出奖励曲线与摘要。
- `trace_q_updates.py`：固定成功路径，逐步打印 `Q` 值更新。

核心更新实现如下：

```python
td_target = reward + config.gamma * np.max(q_table[next_state]) * (not terminated)
td_error = td_target - q_table[state, action]
q_table[state, action] += config.alpha * td_error
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`

## 小结

`Q-Learning` 的关键现象是“终点奖励逐轮向前传播”，而非首轮全局收敛。该机制为后续 `SARSA` 与 `n-step` 的目标值差异提供对照基线。

## 继续阅读

- [05-SARSA的时序更新与策略差异](./05-SARSA的时序更新与策略差异.md)
- [02-frozenlake-tabular-q](../experiments/02-frozenlake-tabular-q/README.md)
