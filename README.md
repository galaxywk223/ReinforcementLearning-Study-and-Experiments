# 强化学习学习与实验

这个仓库整理强化学习入门阶段的学习笔记和可运行实验。正文尽量收拢在 `notes/` 里，`experiments/` 主要保留代码和最小运行说明，避免同一个主题被拆成两条阅读线。

## 当前内容

- [notes/README.md](notes/README.md)：主阅读入口，按学习顺序收录正文
- [experiments/README.md](experiments/README.md)：代码目录索引和运行入口
- [assets/figures/](assets/figures)：主笔记里引用的代表结果图

## 推荐阅读顺序

1. [00-环境安装与运行](notes/00-环境安装与运行.md)
2. [01-第一次理解强化学习](notes/01-第一次理解强化学习.md)
3. [02-MDP、回报与Bellman方程](notes/02-MDP、回报与Bellman方程.md)
4. [03-Q-Learning是怎么一步步把Q表学出来的](notes/03-Q-Learning是怎么一步步把Q表学出来的.md)
5. [04-SARSA是怎么用下一步真实动作更新Q表的](notes/04-SARSA是怎么用下一步真实动作更新Q表的.md)
6. [05-MonteCarlo是怎么用整局回报更新动作价值的](notes/05-MonteCarlo是怎么用整局回报更新动作价值的.md)

如果只想找代码和命令，再去看 [experiments/README.md](experiments/README.md)。

## 主题与代码对应

| 主笔记 | 配套实验目录 | 环境 | 方法 |
| --- | --- | --- | --- |
| [03-Q-Learning是怎么一步步把Q表学出来的](notes/03-Q-Learning是怎么一步步把Q表学出来的.md) | [01-frozenlake-tabular-q](experiments/01-frozenlake-tabular-q/README.md) | `FrozenLake-v1` | `Tabular Q-Learning` |
| [04-SARSA是怎么用下一步真实动作更新Q表的](notes/04-SARSA是怎么用下一步真实动作更新Q表的.md) | [02-cliffwalking-tabular-sarsa](experiments/02-cliffwalking-tabular-sarsa/README.md) | `CliffWalking-v1` | `Tabular SARSA` |
| [05-MonteCarlo是怎么用整局回报更新动作价值的](notes/05-MonteCarlo是怎么用整局回报更新动作价值的.md) | [03-blackjack-monte-carlo](experiments/03-blackjack-monte-carlo/README.md) | `Blackjack-v1` | `First-Visit Monte Carlo Control` |

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

这些结果的解释都已经放回对应主笔记里；这里保留一个总览。

### Q-Learning / FrozenLake

`FrozenLake-v1` 上的表格型 `Q-Learning` 结果可以直接看到奖励曲线逐步抬升。

![FrozenLake reward curve](assets/figures/frozenlake/reward_curve.png)

| 运行名 | 回合数 | 平均奖励 | 成功率 |
| --- | ---: | ---: | ---: |
| `first-full-run` | 4000 | `0.73` | `0.73` |

对应主笔记：[03-Q-Learning是怎么一步步把Q表学出来的](notes/03-Q-Learning是怎么一步步把Q表学出来的.md)

### SARSA / CliffWalking

`SARSA` 基线实验里，代表性策略保持了 `0` 次平均掉崖，并以较长但稳定的路径到达终点。

![CliffWalking reward curve](assets/figures/cliffwalking/reward_curve.png)

| 运行名 | 回合数 | 平均奖励 | 平均到达步数 | 平均掉崖次数 |
| --- | ---: | ---: | ---: | ---: |
| `sarsa-baseline` | 800 | `-17.0` | `17.0` | `0.0` |

对应主笔记：[04-SARSA是怎么用下一步真实动作更新Q表的](notes/04-SARSA是怎么用下一步真实动作更新Q表的.md)

### Monte Carlo / Blackjack

`Blackjack` 的公开示例保留了训练曲线和最终策略热力图，便于把整局回报与最终策略对应起来。

![Blackjack policy heatmaps](assets/figures/blackjack/policy_heatmaps.png)

| 运行名 | 回合数 | 平均奖励 | 胜率 | 平局率 |
| --- | ---: | ---: | ---: | ---: |
| `monte-carlo-reference-500k` | 500000 | `-0.0413` | `0.4350` | `0.0887` |

对应主笔记：[05-MonteCarlo是怎么用整局回报更新动作价值的](notes/05-MonteCarlo是怎么用整局回报更新动作价值的.md)

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
│  └─ 05-MonteCarlo是怎么用整局回报更新动作价值的.md
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
