# Reinforcement Learning Study and Experiments

一个面向公开读者的强化学习学习型仓库，目标是把“基础概念、代码实现、实验结果、可复现实验路径”整理成可以直接阅读和运行的 GitHub 项目。

## 适合谁

- 想系统入门强化学习的人
- 想先从表格方法理解值函数和 Bellman 更新的人
- 想把“公式理解”和“代码执行”对上的人

## 当前覆盖内容

- 文档：从入门问题、MDP 与 Bellman 方程，到 `Q-Learning`、`SARSA` 和两者在 `CliffWalking` 中的对比
- 项目：两个可直接运行的表格方法教学实验
- 结果展示：一份精选训练曲线和对应结果摘要

## 快速开始

推荐先使用仓库根目录下的公开环境定义：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

然后运行当前示例项目：

```bash
cd projects/frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
```

或者运行新的 `CliffWalking SARSA` 项目：

```bash
cd projects/cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
```

更完整的安装说明见 [docs/setup.md](docs/setup.md)。

## 学习路径

建议按下面顺序阅读和运行：

1. [第一次理解强化学习](docs/getting-started/first-questions.md)
2. [MDP、回报与 Bellman 方程](docs/foundations/mdp-and-bellman.md)
3. [Q-Learning 是怎么一步步把 Q 表学出来的](docs/tabular/q-learning-step-by-step.md)
4. [FrozenLake Tabular Q-Learning 项目说明](projects/frozenlake-tabular-q/README.md)
5. [SARSA 是怎么用“下一步真实动作”更新 Q 表的](docs/tabular/sarsa-step-by-step.md)
6. [SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别](docs/tabular/sarsa-vs-q-learning.md)
7. [CliffWalking Tabular SARSA 项目说明](projects/cliffwalking-tabular-sarsa/README.md)

## 项目概览

### FrozenLake Tabular Q-Learning

项目路径：[projects/frozenlake-tabular-q/README.md](projects/frozenlake-tabular-q/README.md)

- 环境：`FrozenLake-v1`
- 方法：`Tabular Q-Learning`
- 目标：理解动作价值表如何被逐步学习出来
- 输出：奖励曲线、评估成功率、最终策略和 Q 表

### CliffWalking Tabular SARSA

项目路径：[projects/cliffwalking-tabular-sarsa/README.md](projects/cliffwalking-tabular-sarsa/README.md)

- 环境：`CliffWalking-v1`
- 方法：`Tabular SARSA`
- 目标：理解 `on-policy` 更新为什么会让策略更保守
- 输出：奖励曲线、平均回报、平均步数、平均掉崖次数、最终策略和 Q 表

## 精选结果

下面的奖励曲线来自当前仓库的一个代表性运行结果：

![FrozenLake reward curve](assets/examples/frozenlake/reward_curve.png)

对应的结果摘要：

| Run | Episodes | Environment | Avg Reward | Success Rate |
| --- | -------: | ----------- | ---------: | -----------: |
| `first-full-run` | 4000 | `FrozenLake-v1` (`is_slippery=True`) | `0.73` | `0.73` |

## 仓库结构

```text
ReinforcementLearning-Study-and-Experiments/
├─ assets/
│  └─ examples/
├─ docs/
│  ├─ getting-started/
│  ├─ foundations/
│  └─ tabular/
├─ projects/
│  ├─ requirements.txt
│  ├─ frozenlake-tabular-q/
│  └─ cliffwalking-tabular-sarsa/
├─ environment.yml
└─ README.md
```

## 路线图

当前仓库已经公开两个表格方法项目。后续会按下面顺序扩展：

1. `FrozenLake` 中不同探索率和环境设置对比
2. `Monte Carlo` 与 `TD` 方法的进一步对比
3. `CartPole` 上的 `DQN`
4. 更完整的策略梯度与 Actor-Critic 实验

## 开源协议

本仓库中的代码与文档基于 [MIT License](./LICENSE) 开源。

补充说明：

- 本仓库的许可证仅覆盖当前仓库中自行编写和整理的代码、说明文档与教学实验组织内容。
- 仓库中的实验结果图、训练输出以及第三方环境、平台和依赖库的原始内容，不因本仓库采用 MIT 协议而自动转授任何额外权利。
