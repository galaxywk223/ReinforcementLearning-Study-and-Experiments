# CartPole `REINFORCE`

这个目录保存 `CartPole-v1` 上的 `REINFORCE` 代码和最小运行说明，用于展示整局回报驱动的策略梯度更新。

## 关联笔记

- [10-REINFORCE的回合策略梯度与高方差问题](../../notes/10-REINFORCE的回合策略梯度与高方差问题.md)

## 实验内容

- 完整训练入口 `train.py`
- 打印回报与损失权重的教学脚本 `trace_reinforce_returns.py`
- 奖励曲线、损失曲线和训练摘要导出

## 代表结果

- 回合数：`400`
- 评估平均回报：`498.0`
- 评估平均回合长度：`498.0`
- 成功率：`0.9`

<p align="center">
  <img src="../../assets/figures/cartpole-reinforce/reward_curve.png" alt="CartPole REINFORCE 奖励曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/08-cartpole-reinforce
python train.py --episodes 400
python train.py --episodes 100 --eval-episodes 10 --run-name smoke
python trace_reinforce_returns.py
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/loss_curve.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `trace_reinforce_returns.py` | 打印回合回报与策略损失权重 |
