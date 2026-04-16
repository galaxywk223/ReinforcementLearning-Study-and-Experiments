# CartPole `Actor-Critic`

这个目录保存 `CartPole-v1` 上的同步 `Actor-Critic` 代码和最小运行说明，用于展示价值基线如何稳定策略梯度更新。

## 关联笔记

- [11-Actor-Critic的价值基线与同步更新](../../notes/11-Actor-Critic的价值基线与同步更新.md)

## 实验内容

- 完整训练入口 `train.py`
- 与 `REINFORCE` 的对比脚本 `compare_actor_critic_reinforce.py`
- 奖励曲线、critic 损失曲线和训练摘要导出

## 代表结果

- 回合数：`400`
- 评估平均回报：`500.0`
- 评估平均回合长度：`500.0`
- 成功率：`1.0`

<p align="center">
  <img src="../../assets/figures/cartpole-actor-critic/comparison_reward_curve.png" alt="CartPole Actor-Critic 与 REINFORCE 对比曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/09-cartpole-actor-critic
python train.py --episodes 400
python compare_actor_critic_reinforce.py
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/critic_loss_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `compare_actor_critic_reinforce.py` | 和 `REINFORCE` 的对比脚本 |
