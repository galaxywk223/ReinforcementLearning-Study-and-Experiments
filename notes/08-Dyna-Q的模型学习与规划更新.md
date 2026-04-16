# Dyna-Q的模型学习与规划更新

本章补齐“无模型表格更新”和“显式使用环境模型”之间的桥：`Dyna-Q` 先从真实交互中学习近似模型，再用规划更新把同一批经验重复利用。

## 本章目标

- 说明 `Dyna-Q` 中真实交互更新与规划更新的分工。
- 观察规划步数如何改变 `CliffWalking` 的样本效率。
- 建立 `Dyna-Q` 与后续深度值方法之间的联系与边界。

## 本章实验

- 主环境：`CliffWalking-v1`
- 方法：表格型 `Dyna-Q`
- 关键配置：每个真实环境步后追加 `10` 次规划更新

## 关键结果

当前仓库基线结果如下：

- 回合数：`400`
- 评估平均回报：`-13.0`
- 平均到达步数：`13.0`
- 平均掉崖次数：`0.0`
- 对比训练后段平均回报：`Dyna-Q=-37.81`，`Q-Learning=-54.90`

<p align="center">
  <img src="../assets/figures/cliffwalking-dyna-q/comparison_reward_curve.png" alt="CliffWalking Dyna-Q 与 Q-Learning 对比曲线" width="920" />
</p>

## 核心机制

### 真实交互更新

`Dyna-Q` 的真实环境更新仍采用一步 `Q-Learning` 目标：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
$$

该部分负责吸收新的状态转移与奖励信息。

### 模型学习

每访问一次状态动作对 $(s,a)$，当前实现都会把其结果写入一个最小模型表：

$$
(s,a)\mapsto (r,s',done)
$$

模型不是环境真值，而是“当前已观察到的经验索引”。

### 规划更新

真实步完成后，从已记录的状态动作对中随机采样若干条，再重复执行同样的 Bellman 更新。这样单条真实经验不仅更新一次，还可在无新交互的情况下被多次回放。

### 为什么最终路径可能相同

在当前 `CliffWalking` 基线上，`Q-Learning` 与 `Dyna-Q` 最终都学到了沿悬崖上边界的最短安全路径。区别主要体现在训练节奏：`Dyna-Q` 更快把少量成功经验传播回更早状态。

## 代码与脚本

### 代码入口

- [train.py](../experiments/06-cliffwalking-dyna-q/train.py)
- [compare_dyna_q_q_learning.py](../experiments/06-cliffwalking-dyna-q/compare_dyna_q_q_learning.py)
- [trace_dyna_q_updates.py](../experiments/06-cliffwalking-dyna-q/trace_dyna_q_updates.py)

### 运行命令

```bash
cd experiments/06-cliffwalking-dyna-q
python train.py --episodes 400 --planning-steps 10 --render-final-policy
python compare_dyna_q_q_learning.py
python trace_dyna_q_updates.py --planning-steps 5
```

### 脚本说明

- `train.py`：完整训练、评估与曲线导出。
- `compare_dyna_q_q_learning.py`：对比有无规划更新时的训练曲线。
- `trace_dyna_q_updates.py`：打印一次真实交互后的多次规划回放。

规划更新的核心结构如下：

```python
sampled_reward, sampled_next_state, sampled_done = model[(sampled_state, sampled_action)]
planned_target = q_learning_target(q_table, sampled_reward, sampled_next_state, sampled_done, config.gamma)
q_table[sampled_state, sampled_action] += config.alpha * (planned_target - q_table[sampled_state, sampled_action])
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 小结

`Dyna-Q` 的关键改造不是换掉值函数更新公式，而是把真实经验转成“可反复规划的模型条目”。该思想在后续深度方法中仍然存在，只是模型表示与规划方式会变得更复杂。

## 继续阅读

- [09-DQN的经验回放与目标网络](./09-DQN的经验回放与目标网络.md)
- [06-cliffwalking-dyna-q](../experiments/06-cliffwalking-dyna-q/README.md)
