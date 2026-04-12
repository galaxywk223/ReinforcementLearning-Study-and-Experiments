# CliffWalking 表格型 SARSA

这个实验用 `CliffWalking-v1` 演示 `SARSA` 在带风险的环境里如何学习动作价值，并和 `Q-Learning` 做直接对比。

## 这个实验在回答什么问题

- `SARSA` 为什么要看下一步真实动作
- `on-policy` 更新会怎样影响最终策略
- 在悬崖环境里，安全路径和最短路径为什么会分开

## 环境与方法

- 环境：`CliffWalking-v1`
- 方法：`Tabular SARSA`
- 动作空间：`U / R / D / L`
- 输出：训练曲线、评估回报、平均步数、平均掉崖次数、最终策略、Q 表

## 运行方式

训练一个基线：

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
```

常用命令：

```bash
python train.py --episodes 1200 --alpha 0.5 --gamma 0.99 --epsilon-start 0.1 --epsilon-end 0.1 --epsilon-decay 1.0
python train.py --episodes 800 --max-steps-per-episode 500
python trace_sarsa_updates.py --episodes 2
python compare_sarsa_q_learning.py --episodes 800
```

`compare_sarsa_q_learning.py` 会保存两种算法的训练曲线和评估摘要，适合直接看对比。

## 输出文件

训练输出位于 `outputs/<run_name>/`：

- `summary.json`
- `reward_curve.png`

对比输出位于 `outputs/comparisons/<run_name>/`：

- `comparison_summary.json`
- `comparison_reward_curve.png`

## 代表性结果

`SARSA` 基线实验在 800 个 episode 后，评估阶段保持了 `0.0` 次平均掉崖，并以平均 `17` 步到达终点。

![CliffWalking reward curve](../../assets/figures/cliffwalking/reward_curve.png)

| 运行名 | 回合数 | 平均奖励 | 平均到达步数 | 平均掉崖次数 |
| --- | ---: | ---: | ---: | ---: |
| `sarsa-baseline` | 800 | `-17.0` | `17.0` | `0.0` |

## 对应笔记

- [04-SARSA是怎么用下一步真实动作更新Q表的](../../notes/04-SARSA是怎么用下一步真实动作更新Q表的.md)
- [05-SARSA和Q-Learning在CliffWalking里会学出什么区别](../../notes/05-SARSA和Q-Learning在CliffWalking里会学出什么区别.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](../../notes/03-Q-Learning是怎么一步步把Q表学出来的.md)
- [00-环境安装与运行](../../notes/00-环境安装与运行.md)
