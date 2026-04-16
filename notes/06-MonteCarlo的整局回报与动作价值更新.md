# MonteCarlo的整局回报与动作价值更新

本章讨论 `Monte Carlo Control` 的更新逻辑：整局结束后再计算回报，并对状态动作价值进行样本平均更新。

## 本章目标

- 区分 Monte Carlo 与时序差分方法的目标值来源。
- 理解 `First-Visit` 约束在更新中的作用。
- 观察整局回报统计如何塑造 `Blackjack` 策略边界。

## 本章实验

- 主环境：`Blackjack-v1`
- 方法：`First-Visit Monte Carlo Control`
- 产物：奖励曲线、策略热力图、价值热力图

## 关键结果

当前仓库基线结果如下：

- 回合数：`500000`
- 评估平均回报：`-0.0413`
- 胜率：`0.4350`
- 平局率：`0.0887`
- 负率：`0.4763`

<p align="center">
  <img src="../assets/figures/blackjack/policy_heatmaps.png" alt="Blackjack Monte Carlo 策略热力图" width="920" />
</p>

## 核心机制

### 与时序差分的核心区别

`Q-Learning` / `SARSA` 在每一步使用自举目标，Monte Carlo 在回合结束后使用整局真实回报：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

### `First-Visit` 更新

同一回合中，对同一 `(state, action)` 仅在首次出现位置更新：

$$
Q(s, a) \leftarrow Q(s, a) + \frac{1}{N}\left(G - Q(s, a)\right)
$$

其中 $N$ 为该状态动作对累计访问次数。

### 为什么使用 `Blackjack`

`Blackjack` 为回合制任务，终局奖励主导价值判断，适合展示“整局结果反向分配到早期动作”的更新机制。

## 代码与脚本

### 代码入口

- [train.py](../experiments/04-blackjack-monte-carlo/train.py)
- [trace_mc_updates.py](../experiments/04-blackjack-monte-carlo/trace_mc_updates.py)

### 运行命令

```bash
cd experiments/04-blackjack-monte-carlo
python train.py --episodes 200000 --render-final-policy
python trace_mc_updates.py --episodes 3
```

### 脚本说明

- `train.py`：完整训练与评估，导出奖励曲线、策略图与价值图。
- `trace_mc_updates.py`：复现固定回合，逐步打印整局回报与首次访问更新。

回报反向计算如下：

```python
for index in range(len(episode) - 1, -1, -1):
    _, _, reward = episode[index]
    discounted_return = reward + config.gamma * discounted_return
    returns[index] = discounted_return
```

首次访问样本平均更新如下：

```python
visit_counts[table_index] += 1
step_size = 1.0 / visit_counts[table_index]
q_table[table_index] += step_size * (returns[index] - q_table[table_index])
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/policy_heatmaps.png`
- `outputs/<run_name>/value_heatmaps.png`

## 小结

Monte Carlo 通过整局回报直接更新动作价值，适合解释终局反馈主导的决策任务。与 TD 方法相比，其差异核心在于“是否依赖下一状态估计值自举”。

## 继续阅读

- [07-n-step-SARSA的多步回报与折中更新](./07-n-step-SARSA的多步回报与折中更新.md)
- [04-blackjack-monte-carlo](../experiments/04-blackjack-monte-carlo/README.md)
