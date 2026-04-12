# 强化学习学习与实验

这个仓库整理强化学习入门阶段的学习笔记和可运行实验。当前内容集中在离散环境与表格方法，笔记和实验一一对应，适合按顺序阅读，也可以直接进入某个实验单独运行。

## 当前内容

- [notes/](notes/README.md)：按序号组织的学习笔记，从基础概念到 `Q-Learning`、`SARSA`、`Monte Carlo`
- [experiments/](experiments/README.md)：三个可运行实验，覆盖 `FrozenLake`、`CliffWalking` 和 `Blackjack`
- [assets/figures/](assets/figures)：README 中引用的精选结果图

## 推荐阅读顺序

1. [00-环境安装与运行](notes/00-环境安装与运行.md)
2. [01-第一次理解强化学习](notes/01-第一次理解强化学习.md)
3. [02-MDP、回报与Bellman方程](notes/02-MDP、回报与Bellman方程.md)
4. [03-Q-Learning是怎么一步步把Q表学出来的](notes/03-Q-Learning是怎么一步步把Q表学出来的.md)
5. [FrozenLake 表格型 Q 学习实验](experiments/01-frozenlake-tabular-q/README.md)
6. [04-SARSA是怎么用下一步真实动作更新Q表的](notes/04-SARSA是怎么用下一步真实动作更新Q表的.md)
7. [05-SARSA和Q-Learning在CliffWalking里会学出什么区别](notes/05-SARSA和Q-Learning在CliffWalking里会学出什么区别.md)
8. [CliffWalking 表格型 SARSA 实验](experiments/02-cliffwalking-tabular-sarsa/README.md)
9. [06-MonteCarlo是怎么用整局回报更新动作价值的](notes/06-MonteCarlo是怎么用整局回报更新动作价值的.md)
10. [Blackjack 首次访问蒙特卡洛实验](experiments/03-blackjack-monte-carlo/README.md)

完整导航见 [notes/README.md](notes/README.md) 和 [experiments/README.md](experiments/README.md)。

## 实验概览

| 实验 | 环境 | 方法 | 主要观察点 |
| --- | --- | --- | --- |
| [FrozenLake 表格型 Q 学习实验](experiments/01-frozenlake-tabular-q/README.md) | `FrozenLake-v1` | `Tabular Q-Learning` | 终点奖励如何沿路径逐步向前传播 |
| [CliffWalking 表格型 SARSA 实验](experiments/02-cliffwalking-tabular-sarsa/README.md) | `CliffWalking-v1` | `Tabular SARSA` | `on-policy` 更新怎样影响风险偏好 |
| [Blackjack 首次访问蒙特卡洛实验](experiments/03-blackjack-monte-carlo/README.md) | `Blackjack-v1` | `First-Visit Monte Carlo Control` | 整局回报如何在回合结束后更新动作价值 |

## 快速开始

推荐使用仓库根目录的环境定义：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

如果不使用 `conda`：

```bash
pip install -r requirements.txt
```

运行第一个实验：

```bash
cd experiments/01-frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
```

更多运行方式见 [00-环境安装与运行](notes/00-环境安装与运行.md)。

## 精选结果

### FrozenLake 实验

`FrozenLake-v1` 上的表格型 `Q-Learning` 结果可以直接看到奖励曲线逐步抬升。

![FrozenLake reward curve](assets/figures/frozenlake/reward_curve.png)

| 运行名 | 回合数 | 平均奖励 | 成功率 |
| --- | ---: | ---: | ---: |
| `first-full-run` | 4000 | `0.73` | `0.73` |

### CliffWalking 实验

`SARSA` 基线实验里，代表性策略保持了 `0` 次平均掉崖，并以较长但稳定的路径到达终点。

![CliffWalking reward curve](assets/figures/cliffwalking/reward_curve.png)

| 运行名 | 回合数 | 平均奖励 | 平均到达步数 | 平均掉崖次数 |
| --- | ---: | ---: | ---: | ---: |
| `sarsa-baseline` | 800 | `-17.0` | `17.0` | `0.0` |

### Blackjack 实验

`Blackjack` 的公开示例保留了训练曲线和最终策略热力图，便于把整局回报与最终策略对应起来。

![Blackjack policy heatmaps](assets/figures/blackjack/policy_heatmaps.png)

| 运行名 | 回合数 | 平均奖励 | 胜率 | 平局率 |
| --- | ---: | ---: | ---: | ---: |
| `monte-carlo-reference-500k` | 500000 | `-0.0413` | `0.4350` | `0.0887` |

## 仓库结构

```text
ReinforcementLearning-Study-and-Experiments/
├─ assets/
│  └─ figures/
├─ experiments/
│  ├─ README.md
│  ├─ 01-frozenlake-tabular-q/
│  ├─ 02-cliffwalking-tabular-sarsa/
│  └─ 03-blackjack-monte-carlo/
├─ notes/
│  ├─ README.md
│  ├─ 00-环境安装与运行.md
│  ├─ 01-第一次理解强化学习.md
│  ├─ 02-MDP、回报与Bellman方程.md
│  ├─ 03-Q-Learning是怎么一步步把Q表学出来的.md
│  ├─ 04-SARSA是怎么用下一步真实动作更新Q表的.md
│  ├─ 05-SARSA和Q-Learning在CliffWalking里会学出什么区别.md
│  └─ 06-MonteCarlo是怎么用整局回报更新动作价值的.md
├─ environment.yml
├─ requirements.txt
└─ README.md
```

## 后续更新

后续内容会继续沿笔记序号往后扩展，优先补下面几条：

1. `TD(0)`、`Expected SARSA` 和 `n-step` 方法
2. `Monte Carlo` 与 `TD` 方法的对比
3. `CartPole` 上的 `DQN`
4. 更完整的策略梯度与 Actor-Critic 实验

## 开源协议

仓库中的代码和文档基于 [MIT License](LICENSE) 开源。
