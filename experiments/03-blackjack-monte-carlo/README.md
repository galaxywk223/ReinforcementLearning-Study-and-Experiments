# Blackjack 首次访问蒙特卡洛

这个目录保存 `Blackjack-v1` 上的首次访问 Monte Carlo Control 代码和最小运行说明，用于展示整局回报如何逐步塑造最终策略边界。

## 关联笔记

- [05-MonteCarlo的整局回报与动作价值更新](../../notes/05-MonteCarlo的整局回报与动作价值更新.md)

## 实验内容

- 完整训练入口 `train.py`
- 逐局打印回报传播过程的 `trace_mc_updates.py`
- 策略热力图、价值热力图和奖励曲线导出

## 代表结果

- 回合数：`500000`
- 评估平均回报：`-0.0413`
- 胜率：`0.4350`
- 平局率：`0.0887`

<p align="center">
  <img src="../../assets/figures/blackjack/policy_heatmaps.png" alt="Blackjack Monte Carlo 策略热力图" width="920" />
</p>

## 运行命令

```bash
cd experiments/03-blackjack-monte-carlo
python train.py --episodes 200000 --render-final-policy
python train.py --episodes 300000 --epsilon-start 0.2 --epsilon-end 0.03 --epsilon-decay 0.999992 --run-name blackjack-mc-300k
python train.py --episodes 80000 --epsilon-start 0.15 --epsilon-end 0.05 --epsilon-decay 0.99998 --run-name blackjack-mc-fast
python trace_mc_updates.py --episodes 3
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/policy_heatmaps.png`
- `outputs/<run_name>/value_heatmaps.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `trace_mc_updates.py` | 逐局打印回报传播的回报追踪脚本 |
