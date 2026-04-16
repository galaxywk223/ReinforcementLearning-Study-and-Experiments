# Pendulum `SAC`

这个目录保存 `Pendulum-v1` 上的 `SAC` 代码和最小运行说明，用于展示最大熵目标、双 `Q-network` 与连续动作策略分布的组合方式。

## 关联笔记

- [13-SAC的最大熵目标与连续动作控制](../../notes/13-SAC的最大熵目标与连续动作控制.md)

## 实验内容

- 完整训练入口 `train.py`
- 打印软 `Q` 目标组成项的教学脚本 `trace_sac_targets.py`
- 奖励曲线、critic 损失曲线和训练摘要导出

## 代表结果

- 总环境步数：`10000`
- 评估平均回报：`-178.3184`
- 评估平均回合长度：`200.0`
- 最终温度参数：`0.1173`

<p align="center">
  <img src="../../assets/figures/pendulum-sac/reward_curve.png" alt="Pendulum SAC 奖励曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/11-pendulum-sac
python train.py --total-env-steps 10000
python train.py --total-env-steps 3000 --learning-starts 256 --batch-size 64 --eval-episodes 3 --run-name smoke
python trace_sac_targets.py
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/q_loss_curve.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `trace_sac_targets.py` | 打印软 `Q` 目标组成项 |
