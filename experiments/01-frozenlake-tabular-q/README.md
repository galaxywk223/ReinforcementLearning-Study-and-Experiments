# FrozenLake 表格型 Q 学习

这个实验用 `FrozenLake-v1` 演示表格型 `Q-Learning` 如何从零开始学习动作价值表，并把奖励一轮一轮传回更早的状态。

## 这个实验在回答什么问题

- Q 值一开始是什么样
- 终点奖励为什么不会一次传回整条路径
- 探索率衰减会怎样影响训练表现

## 环境与方法

- 环境：`FrozenLake-v1`
- 方法：`Tabular Q-Learning`
- 动作空间：`L / D / R / U`
- 输出：训练曲线、评估成功率、最终策略、Q 表

## 运行方式

```bash
cd experiments/01-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
```

常用命令：

```bash
python train.py --episodes 6000 --alpha 0.15 --gamma 0.99 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.999
python train.py --episodes 4000 --non-slippery --run-name frozenlake-deterministic
python trace_q_updates.py --episodes 6
```

## 输出文件

训练结束后会在 `outputs/<run_name>/` 下生成：

- `summary.json`
- `reward_curve.png`

`outputs/` 默认不纳入版本控制，README 里引用的是 `assets/figures/` 中保留的示例图。

## 代表性结果

![FrozenLake reward curve](../../assets/figures/frozenlake/reward_curve.png)

| Run | Episodes | Avg Reward | Success Rate |
| --- | ---: | ---: | ---: |
| `first-full-run` | 4000 | `0.73` | `0.73` |

## 对应笔记

- [01-第一次理解强化学习](../../notes/01-第一次理解强化学习.md)
- [02-MDP、回报与Bellman方程](../../notes/02-MDP、回报与Bellman方程.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](../../notes/03-Q-Learning是怎么一步步把Q表学出来的.md)
- [00-环境安装与运行](../../notes/00-环境安装与运行.md)
