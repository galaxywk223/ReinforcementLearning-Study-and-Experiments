# CliffWalking Tabular SARSA

一个面向表格方法下一阶段学习的最小完整项目，用 `CliffWalking-v1` 演示 `SARSA` 如何在带风险的环境中学习更保守的动作价值表。

## 项目目标

- 理解 `SARSA` 更新时为什么要看下一步真实动作
- 观察 `on-policy` 学习在危险环境中的表现
- 对比 `SARSA` 和 `Q-Learning` 在 `CliffWalking` 里的路径差异

## 环境与算法

- 环境：`CliffWalking-v1`
- 算法：`Tabular SARSA`
- 动作空间：离散动作 `U / R / D / L`
- 输出内容：训练曲线、评估回报、平均步数、平均掉崖次数、最终策略和 Q 表

## 运行方式

在仓库根目录准备环境后，执行：

```bash
cd projects/cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
```

如需安装最小依赖：

```bash
pip install -r ../requirements.txt
```

## 常用命令

训练一个 `SARSA` 基线：

```bash
python train.py --episodes 1200 --alpha 0.5 --gamma 0.99 --epsilon-start 0.1 --epsilon-end 0.1 --epsilon-decay 1.0
```

查看固定安全路径上的更新过程：

```bash
python trace_sarsa_updates.py --episodes 2
```

直接对比 `SARSA` 和 `Q-Learning`：

```bash
python compare_sarsa_q_learning.py --episodes 800
```

这个对比脚本除了打印两张 `greedy policy` 表，还会额外打印：

- 从起点出发的实际贪心 rollout

这样可以直接区分“整张策略表”和“真正走出来的路径”。

如果想限制每回合最长步数，避免早期策略在网格里无限打转：

```bash
python train.py --episodes 800 --max-steps-per-episode 500
```

## 输出文件

训练完成后会在 `outputs/<run_name>/` 下生成：

- `summary.json`：训练参数、评估结果、最终策略和 Q 表
- `reward_curve.png`：平滑后的训练回报曲线

对比脚本会在 `outputs/comparisons/<run_name>/` 下生成：

- `comparison_summary.json`：两种算法的训练与评估摘要
- `comparison_reward_curve.png`：两种算法的训练曲线对比图

`outputs/` 目录默认不会纳入版本控制。

## 你最应该观察什么

这个项目里最值得看的不只是“到没到终点”，而是：

- 平均每回合回报
- 到达终点平均要走多少步
- 每回合平均掉下悬崖多少次
- 最终策略是否贴着悬崖边走

因为 `CliffWalking` 的关键不是“能不能到”，而是“到达过程中承担了多大风险”。

## 三个容易混淆的点

### `trace` 脚本和 `compare` 脚本用途不同

- `trace_sarsa_updates.py` 使用的是固定安全路径，只负责演示 `SARSA` 的更新公式
- `compare_sarsa_q_learning.py` 才是真正让算法自己训练并比较最终策略

所以两边看到的路径或数值不同是正常的，不表示实现冲突。

### `greedy policy` 表不是一条完整路径

打印出来的策略表表示的是：

- 每个状态各自最推荐的动作

真正的贪心路径，需要从起点开始，一步步按这张表往下走，直到终点或进入循环。

### 这个项目会强制限制每回合步数

`CliffWalking-v1` 默认没有官方步数上限，早期策略可能会一直在网格里绕。

因此当前项目在训练和评估时都加了：

- `max_steps_per_episode`

默认值是 `500`，目的是防止实验看起来像卡死。

## 相关文档

- [SARSA 是怎么用“下一步真实动作”更新 Q 表的](../../docs/tabular/sarsa-step-by-step.md)
- [SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别](../../docs/tabular/sarsa-vs-q-learning.md)
- [Q-Learning 是怎么一步步把 Q 表学出来的](../../docs/tabular/q-learning-step-by-step.md)
- [环境安装说明](../../docs/setup.md)
