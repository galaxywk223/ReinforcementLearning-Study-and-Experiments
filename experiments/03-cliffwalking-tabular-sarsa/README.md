# CliffWalking 表格型 SARSA

这个目录保存 `CliffWalking-v1` 上的表格型 `SARSA` 代码和最小运行说明，用于展示风险敏感的 on-policy 更新如何影响策略选择。

## 关联笔记

- [05-SARSA的时序更新与策略差异](../../notes/05-SARSA的时序更新与策略差异.md)

## 实验内容

- 完整训练入口 `train.py`
- 固定安全路径的更新追踪脚本 `trace_sarsa_updates.py`
- 与 `Q-Learning` 的对比脚本 `compare_sarsa_q_learning.py`

## 代表结果

- 回合数：`800`
- 评估平均回报：`-17.0`
- 平均到达步数：`17.0`
- 平均掉崖次数：`0.0`

<p align="center">
  <img src="../../assets/figures/cliffwalking/reward_curve.png" alt="CliffWalking SARSA 奖励曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/03-cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
python train.py --episodes 1200 --alpha 0.5 --gamma 0.99 --epsilon-start 0.1 --epsilon-end 0.1 --epsilon-decay 1.0
python train.py --episodes 800 --max-steps-per-episode 500
python trace_sarsa_updates.py --episodes 2
python compare_sarsa_q_learning.py --episodes 800
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 完整训练入口 |
| `trace_sarsa_updates.py` | 固定安全路径的更新追踪脚本 |
| `compare_sarsa_q_learning.py` | 和 `Q-Learning` 的对比脚本 |
