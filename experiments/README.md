# 实验索引

`experiments/` 保存实验代码与最小运行说明。[notes/README.md](../notes/README.md) 对应概念主线，各实验目录提供具体脚本入口。

## 当前实验

| 目录 | 主笔记 | 主入口 | 说明 |
| --- | --- | --- | --- |
| [01-frozenlake-tabular-q](./01-frozenlake-tabular-q/README.md) | [03-Q-Learning](../notes/03-Q-Learning的值传播与Q表更新.md) | `python train.py --episodes 4000 --render-final-policy` | `FrozenLake-v1` 上的表格型 `Q-Learning` |
| [02-cliffwalking-tabular-sarsa](./02-cliffwalking-tabular-sarsa/README.md) | [04-SARSA](../notes/04-SARSA的时序更新与策略差异.md) | `python train.py --episodes 800 --render-final-policy` | `CliffWalking-v1` 上的表格型 `SARSA` |
| [03-blackjack-monte-carlo](./03-blackjack-monte-carlo/README.md) | [05-MonteCarlo](../notes/05-MonteCarlo的整局回报与动作价值更新.md) | `python train.py --episodes 200000 --render-final-policy` | `Blackjack-v1` 上的首次访问 Monte Carlo Control |
| [04-cliffwalking-n-step-sarsa](./04-cliffwalking-n-step-sarsa/README.md) | [06-n-step SARSA](../notes/06-n-step-SARSA的多步回报与折中更新.md) | `python train.py --episodes 800 --n-step 4 --render-final-policy` | `CliffWalking-v1` 上的表格型 `4-step SARSA` |
| [05-cartpole-dqn](./05-cartpole-dqn/README.md) | [07-DQN](../notes/07-DQN的经验回放与目标网络.md) | `python train.py --episodes 400 --print-eval-rollout` | `CartPole-v1` 上的 vanilla `DQN` |

## 运行前准备

仓库根目录命令如下：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

非 `conda` 安装方式如下：

```bash
pip install -r requirements.txt
```

更完整的环境说明和命令解释见 [00-环境安装与运行](../notes/00-环境安装与运行.md)。
