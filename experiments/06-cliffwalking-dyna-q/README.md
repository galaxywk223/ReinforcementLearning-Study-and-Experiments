# CliffWalking 表格型 Dyna-Q

这个目录保存 `CliffWalking-v1` 上的表格型 `Dyna-Q` 代码和最小运行说明，用于展示模型学习与规划更新如何提升样本利用率。

## 关联笔记

- [08-Dyna-Q的模型学习与规划更新](../../notes/08-Dyna-Q的模型学习与规划更新.md)

## 实验内容

- 完整训练入口 `train.py`
- 与 `Q-Learning` 的对比脚本 `compare_dyna_q_q_learning.py`
- 打印一次真实交互后多次规划更新的教学脚本 `trace_dyna_q_updates.py`

## 代表结果

- 回合数：`400`
- 评估平均回报：`-13.0`
- 平均到达步数：`13.0`
- 平均掉崖次数：`0.0`

<p align="center">
  <img src="../../assets/figures/cliffwalking-dyna-q/comparison_reward_curve.png" alt="CliffWalking Dyna-Q 与 Q-Learning 对比曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/06-cliffwalking-dyna-q
python train.py --episodes 400 --planning-steps 10 --render-final-policy
python compare_dyna_q_q_learning.py
python trace_dyna_q_updates.py --planning-steps 5
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
| `compare_dyna_q_q_learning.py` | 和 `Q-Learning` 的对比脚本 |
| `trace_dyna_q_updates.py` | 打印一次真实步后的规划回放 |
