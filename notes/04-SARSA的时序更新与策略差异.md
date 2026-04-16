# SARSA的时序更新与策略差异

本章分析 `SARSA` 与 `Q-Learning` 的核心差异：更新目标是否显式纳入“下一步真实会执行的动作”。

## 本章目标

- 理解 `SARSA` 更新目标与 `Q-Learning` 的差别。
- 解释 `CliffWalking` 中“更保守路径偏好”的来源。
- 对齐固定路径追踪现象与完整训练评估结果。

## 本章实验

- 主环境：`CliffWalking-v1`
- 对比对象：表格型 `SARSA` 与表格型 `Q-Learning`
- 辅助脚本：固定安全路径追踪、双算法训练对比

## 关键结果

当前仓库 `SARSA` 基线结果如下：

- 回合数：`800`
- 评估平均回报：`-17.0`
- 平均到达步数：`17.0`
- 平均掉崖次数：`0.0`

<p align="center">
  <img src="../assets/figures/cliffwalking/reward_curve.png" alt="CliffWalking SARSA 奖励曲线" width="920" />
</p>

## 核心机制

### 更新目标差异

`Q-Learning` 目标值：

$$
r + \gamma \max_{a'} Q(s', a')
$$

`SARSA` 目标值：

$$
r + \gamma Q(s', a')
$$

其中 `(s, a, r, s', a')` 表示当前动作后的真实下一动作序列。

### 一个最小数值例子

若下一状态动作价值为 `Up=10, Right=7, Down=3, Left=-5`，则：

- `Q-Learning` 始终使用 `10`
- `SARSA` 使用实际采样到的 `a'` 对应值

当 `a'=Left` 时，若 `r=-1, \gamma=0.9`，目标值为：

$$
-1 + 0.9 \times (-5) = -5.5
$$

该机制会把探索风险显式写入当前更新。

### 路径偏好解释

在 `CliffWalking` 中，最短路径靠近悬崖，探索期存在高代价偏离风险。`SARSA` 因纳入实际后续动作，更易形成远离悬崖的保守策略。

## 代码与脚本

### 代码入口

- [train.py](../experiments/02-cliffwalking-tabular-sarsa/train.py)
- [trace_sarsa_updates.py](../experiments/02-cliffwalking-tabular-sarsa/trace_sarsa_updates.py)
- [compare_sarsa_q_learning.py](../experiments/02-cliffwalking-tabular-sarsa/compare_sarsa_q_learning.py)

### 运行命令

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
python trace_sarsa_updates.py --episodes 3
python compare_sarsa_q_learning.py --episodes 800
```

### 脚本说明

- `train.py`：单算法训练与评估，输出策略与回报指标。
- `trace_sarsa_updates.py`：固定路径，展示 `SARSA` 的时序更新过程。
- `compare_sarsa_q_learning.py`：并行训练两算法并汇总对比曲线与指标。

核心更新实现如下：

```python
next_action = epsilon_greedy_action(q_table, next_state, epsilon, env.action_space, rng)
td_target = reward + config.gamma * q_table[next_state, next_action]
td_error = td_target - q_table[state, action]
q_table[state, action] += config.alpha * td_error
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 小结

`SARSA` 的核心价值在于把真实策略行为写入目标值，使路径选择同时考虑期望回报与探索风险。该性质在高惩罚边界环境中更容易观察到。

## 继续阅读

- [05-MonteCarlo的整局回报与动作价值更新](./05-MonteCarlo的整局回报与动作价值更新.md)
- [06-n-step-SARSA的多步回报与折中更新](./06-n-step-SARSA的多步回报与折中更新.md)
- [02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md)
