# FrozenLake 表格型 Q 学习

这个目录只放 `FrozenLake-v1` 上的表格型 `Q-Learning` 代码和最小运行说明。

主说明请看：

- [03-Q-Learning的值传播与Q表更新](../../notes/03-Q-Learning的值传播与Q表更新.md)

## 常用命令

```bash
cd experiments/01-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
python train.py --episodes 6000 --alpha 0.15 --gamma 0.99 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.999
python train.py --episodes 4000 --non-slippery --run-name frozenlake-deterministic
python trace_q_updates.py --episodes 6
```

## 输出文件

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/reward_curve.png`

## 目录文件

- `train.py`：完整训练入口
- `trace_q_updates.py`：固定成功路径的更新追踪脚本
