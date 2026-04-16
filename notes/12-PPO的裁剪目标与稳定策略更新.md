# PPO的裁剪目标与稳定策略更新

本章进入当前工程实践中最常见的 on-policy 策略优化主线：通过裁剪概率比值，限制单次策略更新的幅度，在批量采样和多轮优化之间取得稳定折中。

## 本章目标

- 说明 `PPO-Clip` 的目标函数结构。
- 理解 `GAE`、并行采样与多轮 minibatch 更新的配合关系。
- 观察当前 `LunarLander` 基线下“训练稳定但尚未完全着陆”的表现。

## 本章实验

- 主环境：`LunarLander-v3`
- 方法：离散动作 `PPO-Clip`
- 关键配置：`8` 个并行环境、`GAE(\lambda)`、每批 `4` 轮 `minibatch` 更新

## 关键结果

当前仓库基线结果如下：

- 总环境步数：`200000`
- 评估平均回报：`-46.3309`
- 评估平均回合长度：`425.3`
- 成功率：`0.0`

<p align="center">
  <img src="../assets/figures/lunarlander-ppo/reward_curve.png" alt="LunarLander PPO 奖励曲线" width="920" />
</p>

## 核心机制

### 裁剪目标

对旧策略与新策略的概率比值

$$
r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

`PPO-Clip` 使用：

$$
L^{clip}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

核心作用是：当策略更新过大时，直接截断目标收益，避免一次更新把策略推得过远。

### `GAE`

当前实现使用广义优势估计：

$$
\hat{A}_t=\delta_t+\gamma\lambda\delta_{t+1}+\gamma^2\lambda^2\delta_{t+2}+\cdots
$$

它在偏差和方差之间做折中，比单步 `TD error` 更稳定，也比整局回报更平滑。

### 并行采样与多轮优化

- 先在 `8` 个环境上同步采样一个 rollout batch。
- 再把同一批样本打散为多个 minibatch，重复优化若干轮。

这使 `PPO` 同时具备 on-policy 方法的目标一致性与较好的计算利用率。

### 当前基线的解释

当前 `200k` 步基线已经把平均回合长度拉高到 `425.3`，说明策略学到了更长时间的姿态控制。但平均回报仍为负值，表明“稳定着陆”尚未形成；该结果用于展示 `PPO` 训练稳定化机制，而不是追求环境满分。

## 代码与脚本

### 代码入口

- [train.py](../experiments/10-lunarlander-ppo/train.py)
- [trace_ppo_clipping.py](../experiments/10-lunarlander-ppo/trace_ppo_clipping.py)

### 运行命令

```bash
cd experiments/10-lunarlander-ppo
python train.py --total-env-steps 200000
python train.py --total-env-steps 4096 --num-envs 4 --rollout-steps 128 --update-epochs 2 --minibatch-size 128 --eval-episodes 3 --run-name smoke
python trace_ppo_clipping.py
```

### 脚本说明

- `train.py`：完整训练、评估与曲线导出。
- `trace_ppo_clipping.py`：打印裁剪前后的 surrogate objective。

核心裁剪目标如下：

```python
ratio = torch.exp(new_log_probs - batch_log_probs[minibatch_tensor])
unclipped_objective = ratio * batch_advantages[minibatch_tensor]
clipped_objective = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch_advantages[minibatch_tensor]
policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
```

### 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/loss_curve.png`

## 小结

`PPO` 的关键贡献不在“更换策略梯度方向”，而在“限制每次策略变化的幅度”。该思想解释了为什么现代 on-policy 方法能在更复杂环境中保持较强稳定性。

## 继续阅读

- [13-SAC的最大熵目标与连续动作控制](./13-SAC的最大熵目标与连续动作控制.md)
- [10-lunarlander-ppo](../experiments/10-lunarlander-ppo/README.md)
