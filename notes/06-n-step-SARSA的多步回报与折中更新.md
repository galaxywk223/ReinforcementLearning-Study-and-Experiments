# n-step SARSA的多步回报与折中更新

本章说明 `n-step SARSA` 如何在单步 `SARSA` 与整局 `Monte Carlo` 之间形成折中：先累计若干步真实奖励，再追加一个远端自举项。

## 本章目标

- 理解 `n-step return` 的构造方式与边界情况。
- 对比 `1-step` 与 `n-step` 在信用分配节奏上的差异。
- 结合 `CliffWalking` 观察多步目标对路径偏好的影响。

## 本章实验

- 主环境：`CliffWalking-v1`
- 核心设置：`n=4` 的表格型 `n-step SARSA`
- 辅助脚本：固定安全路径追踪、`1-step` vs `n-step` 对比

## 关键结果

当前仓库 `4-step SARSA` 基线结果如下：

- 回合数：`800`
- 评估平均回报：`-19.0`
- 平均到达步数：`19.0`
- 平均掉崖次数：`0.0`

同配置下 `1-step SARSA` 评估平均回报为 `-17.0`。

<p align="center">
  <img src="../assets/figures/cliffwalking-n-step/comparison_reward_curve.png" alt="CliffWalking 1-step 与 4-step SARSA 对比曲线" width="920" />
</p>

## 核心机制

### 多步回报定义

当时间步 `t` 后仍可继续 `n` 步时：

$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

若回合在 `n` 步内结束，自举项消失，仅保留终局前真实奖励。

### 更新关系

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[G_t^{(n)} - Q(s_t, a_t)\right]
$$

当 `n=1` 时退化为 `SARSA`，当终局在 `n` 步内发生时更接近 `Monte Carlo`。

### 最小数值例子

固定安全路径前四步奖励均为 `-1`，当 `n=4`、$\gamma=0.99$ 时：

$$
G_0^{(4)} = -1 - 0.99 - 0.99^2 - 0.99^3 = -3.940399
$$

若 $\alpha=0.5$，则首个更新值：

$$
Q(36, U) \leftarrow -1.9701995
$$

这说明 `4-step` 会更早把多步代价压回起点附近动作。

## 代码与脚本

### 代码入口

- [train.py](../experiments/04-cliffwalking-n-step-sarsa/train.py)
- [trace_n_step_updates.py](../experiments/04-cliffwalking-n-step-sarsa/trace_n_step_updates.py)
- [compare_one_step_n_step_sarsa.py](../experiments/04-cliffwalking-n-step-sarsa/compare_one_step_n_step_sarsa.py)

### 运行命令

```bash
cd experiments/04-cliffwalking-n-step-sarsa
python train.py --episodes 800 --n-step 4 --render-final-policy
python trace_n_step_updates.py --episodes 3 --n-step 4
python compare_one_step_n_step_sarsa.py --episodes 800 --n-step 4
```

### 脚本说明

- `train.py`：`n-step` 训练与评估主入口。
- `trace_n_step_updates.py`：固定路径，展示多步回报如何延迟并放大早期更新。
- `compare_one_step_n_step_sarsa.py`：对比 `1-step` 与 `n-step` 的训练曲线与最终指标。

核心更新实现如下：

```python
for index in range(tau + 1, upper_bound + 1):
    discounted_return += (config.gamma ** (index - tau - 1)) * rewards[index]

if terminal_time is None or tau + config.n_step < terminal_time:
    discounted_return += (
        config.gamma**config.n_step
        * q_table[states[tau + config.n_step], actions[tau + config.n_step]]
    )
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 小结

`n-step SARSA` 改变的不是“是否必然更优”，而是“早期状态何时获得后续代价信息”。该机制直接影响训练节奏与路径偏好。

## 继续阅读

- [07-DQN的经验回放与目标网络](./07-DQN的经验回放与目标网络.md)
- [04-cliffwalking-n-step-sarsa](../experiments/04-cliffwalking-n-step-sarsa/README.md)
