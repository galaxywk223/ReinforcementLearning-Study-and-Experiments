# FrozenLake 表格型 Q 学习（Tabular Q-Learning）

一个最小但完整的强化学习入门项目，用 `FrozenLake-v1` 演示表格型 Q 学习（`Q-Learning`）如何学习动作价值表，并最终导出策略。

## 项目目标

- 在离散小环境中理解 $Q(s, a)$ 的含义
- 观察探索率衰减如何影响训练
- 看到奖励如何通过 Bellman 递推更新逐步向前传播

## 环境与算法

- 环境：`FrozenLake-v1`
- 算法：表格型 Q 学习（`Tabular Q-Learning`）
- 动作空间：离散动作 `L / D / R / U`
- 输出内容：训练曲线、评估成功率、最终策略、Q 表

## 运行方式

在仓库根目录准备环境后，执行：

```bash
cd projects/01-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
```

如需安装最小依赖：

```bash
pip install -r ../requirements.txt
```

## 常用参数

```bash
python train.py --episodes 6000 --alpha 0.15 --gamma 0.99 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.999
python train.py --episodes 4000 --non-slippery --run-name frozenlake-deterministic
```

## 输出文件

训练完成后会在 `outputs/<run_name>/` 下生成：

- `summary.json`：训练参数、评估结果、最终策略和 Q 表
- `reward_curve.png`：平滑后的训练奖励曲线

`outputs/` 目录默认不会纳入版本控制。仓库根目录的展示结果来自精选示例，而不是直接提交整个输出目录。

## 精选结果

代表性结果图：

![FrozenLake reward curve](../../assets/examples/frozenlake/reward_curve.png)

代表性结果摘要：

| 运行名 | 回合数 | 平均奖励 | 成功率 |
| --- | ---: | ---: | ---: |
| `first-full-run` | 4000 | `0.73` | `0.73` |

## 相关文档

- [第一次理解强化学习](../../notes/01-第一次理解强化学习.md)
- [马尔可夫决策过程（MDP）、回报与 Bellman 方程](../../notes/02-MDP、回报与 Bellman 方程.md)
- [Q 学习（Q-Learning）是怎么一步步把 Q 表学出来的](../../notes/03-Q-Learning 是怎么一步步把 Q 表学出来的.md)
- [环境安装说明](../../notes/00-环境安装.md)
