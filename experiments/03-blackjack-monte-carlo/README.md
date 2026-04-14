# Blackjack 首次访问蒙特卡洛

这个目录只放 `Blackjack-v1` 上的首次访问蒙特卡洛控制代码和最小运行说明。

主说明请看：

- [05-MonteCarlo的整局回报与动作价值更新](../../notes/05-MonteCarlo的整局回报与动作价值更新.md)

## 常用命令

```bash
cd experiments/03-blackjack-monte-carlo
python train.py --episodes 200000 --render-final-policy
python train.py --episodes 300000 --epsilon-start 0.2 --epsilon-end 0.03 --epsilon-decay 0.999992 --run-name blackjack-mc-300k
python train.py --episodes 80000 --epsilon-start 0.15 --epsilon-end 0.05 --epsilon-decay 0.99998 --run-name blackjack-mc-fast
python trace_mc_updates.py --episodes 3
```

## 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`
- `outputs/<run_name>/policy_heatmaps.png`
- `outputs/<run_name>/value_heatmaps.png`

## 目录文件

- `train.py`：完整训练入口
- `trace_mc_updates.py`：逐局打印回报传播的回报追踪脚本
