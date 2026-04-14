# CliffWalking 表格型 SARSA

这个目录只放 `CliffWalking-v1` 上的表格型 `SARSA` 代码和最小运行说明。

主说明请看：

- [04-SARSA的时序更新与策略差异](../../notes/04-SARSA的时序更新与策略差异.md)

## 常用命令

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
python train.py --episodes 1200 --alpha 0.5 --gamma 0.99 --epsilon-start 0.1 --epsilon-end 0.1 --epsilon-decay 1.0
python train.py --episodes 800 --max-steps-per-episode 500
python trace_sarsa_updates.py --episodes 2
python compare_sarsa_q_learning.py --episodes 800
```

## 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/comparisons/<run_name>/comparison_summary.json`
- `outputs/comparisons/<run_name>/comparison_reward_curve.png`

## 目录文件

- `train.py`：完整训练入口
- `trace_sarsa_updates.py`：固定安全路径的更新追踪脚本
- `compare_sarsa_q_learning.py`：和 `Q-Learning` 的对比脚本
