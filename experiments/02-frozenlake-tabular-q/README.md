# FrozenLake 表格型 Q 学习

这个目录保存 `FrozenLake-v1` 上的表格型 `Q-Learning` 代码和最小运行说明，用于展示奖励如何沿着成功轨迹向前传播。

## 关联笔记

- [04-Q-Learning的值传播与Q表更新](../../notes/04-Q-Learning的值传播与Q表更新.md)

## 实验内容

- 完整训练入口 `train.py`
- 固定成功路径的更新追踪脚本 `trace_q_updates.py`
- 可切换 `slippery / non-slippery` 的对照运行方式

## 代表结果

- 回合数：`4000`
- 评估平均奖励：`0.73`
- 评估成功率：`0.73`

<p align="center">
  <img src="../../assets/figures/frozenlake/reward_curve.png" alt="FrozenLake Q-Learning 奖励曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/02-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
python train.py --episodes 6000 --alpha 0.15 --gamma 0.99 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.999
python train.py --episodes 4000 --non-slippery --run-name frozenlake-deterministic
python trace_q_updates.py --episodes 6
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `trace_q_updates.py` | 固定成功路径的更新追踪脚本 |
