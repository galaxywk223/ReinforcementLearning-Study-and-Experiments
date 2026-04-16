# SAC的最大熵目标与连续动作控制

本章补齐“连续动作 off-policy 深度强化学习”的主线：在保留 replay buffer 和目标网络的同时，把策略目标从“只追求高回报”改为“高回报 + 高熵”。

## 本章目标

- 说明 `SAC` 的最大熵目标与温度参数含义。
- 理解双 `Q-network`、目标网络与随机策略的协作关系。
- 观察 `Pendulum` 上连续动作控制的最小可运行基线。

## 本章实验

- 主环境：`Pendulum-v1`
- 方法：连续动作 `SAC`
- 关键配置：双 `Q-network`、`tanh` 高斯策略、自动温度调节

## 关键结果

当前仓库基线结果如下：

- 总环境步数：`10000`
- 评估平均回报：`-178.3184`
- 评估平均回合长度：`200.0`
- 最终温度参数：`0.1173`

<p align="center">
  <img src="../assets/figures/pendulum-sac/reward_curve.png" alt="Pendulum SAC 奖励曲线" width="920" />
</p>

## 核心机制

### 最大熵目标

`SAC` 不只优化回报，还鼓励策略保持较高熵：

$$
J(\pi)=\sum_t \mathbb{E}\left[r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot \mid s_t))\right]
$$

其中 $\alpha$ 控制“回报最大化”和“策略随机性”之间的权重。

### 软 `Q` 目标

当前实现的目标值为：

$$
y=r+\gamma\left(\min(Q_1',Q_2')-\alpha \log \pi(a' \mid s')\right)
$$

软价值项既考虑下一步值函数，也把策略熵写进目标。

### 双 `Q-network`

`SAC` 使用两个 critic，并在目标值里取较小者：

$$
\min(Q_1,Q_2)
$$

这样可以抑制单个 critic 的过高估计。

### 为什么切到 `Pendulum`

`Pendulum` 动作为连续标量，表格法和离散动作策略不再适用。该环境用于展示：一旦动作空间从离散切到连续，策略网络必须直接输出分布参数。

## 代码与脚本

### 代码入口

- [train.py](../experiments/11-pendulum-sac/train.py)
- [trace_sac_targets.py](../experiments/11-pendulum-sac/trace_sac_targets.py)

### 运行命令

```bash
cd experiments/11-pendulum-sac
python train.py --total-env-steps 10000
python train.py --total-env-steps 3000 --learning-starts 256 --batch-size 64 --eval-episodes 3 --run-name smoke
python trace_sac_targets.py
```

### 脚本说明

- `train.py`：完整训练、评估与曲线导出。
- `trace_sac_targets.py`：打印软 `Q` 目标中的各个组成项。

核心目标值构造如下：

```python
target_q = torch.min(target_q1(next_states, next_actions), target_q2(next_states, next_actions))
td_target = rewards + config.gamma * (1.0 - dones) * (target_q - alpha * next_log_probs)
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/q_loss_curve.png`

## 小结

`SAC` 把 replay buffer、目标网络、随机策略和最大熵目标组合在一起，形成了连续控制里最常见的一条 off-policy 深度强化学习主线。至此，当前仓库已经覆盖了从动态规划到连续动作深度方法的最小主干闭环。

## 继续阅读

- [experiments/README.md](../experiments/README.md)
- [11-pendulum-sac](../experiments/11-pendulum-sac/README.md)
