# 实验索引

`experiments/` 保存实验代码与最小运行说明。[notes/README.md](../notes/README.md) 对应概念主线，各实验目录提供具体脚本入口。

各笔记章节中的“代码与脚本”段均对应本索引中的实验目录入口。

## 当前实验

| 目录 | 主笔记 | 主入口 | 说明 |
| --- | --- | --- | --- |
| [01-frozenlake-dp](./01-frozenlake-dp/README.md) | [03-动态规划](../notes/03-动态规划的策略评估、策略迭代与价值迭代.md) | `python train.py --render-final-policy` | `FrozenLake-v1` 上的策略迭代与价值迭代 |
| [02-frozenlake-tabular-q](./02-frozenlake-tabular-q/README.md) | [04-Q-Learning](../notes/04-Q-Learning的值传播与Q表更新.md) | `python train.py --episodes 4000 --render-final-policy` | `FrozenLake-v1` 上的表格型 `Q-Learning` |
| [03-cliffwalking-tabular-sarsa](./03-cliffwalking-tabular-sarsa/README.md) | [05-SARSA](../notes/05-SARSA的时序更新与策略差异.md) | `python train.py --episodes 800 --render-final-policy` | `CliffWalking-v1` 上的表格型 `SARSA` |
| [04-blackjack-monte-carlo](./04-blackjack-monte-carlo/README.md) | [06-MonteCarlo](../notes/06-MonteCarlo的整局回报与动作价值更新.md) | `python train.py --episodes 200000 --render-final-policy` | `Blackjack-v1` 上的首次访问 Monte Carlo Control |
| [05-cliffwalking-n-step-sarsa](./05-cliffwalking-n-step-sarsa/README.md) | [07-n-step SARSA](../notes/07-n-step-SARSA的多步回报与折中更新.md) | `python train.py --episodes 800 --n-step 4 --render-final-policy` | `CliffWalking-v1` 上的表格型 `4-step SARSA` |
| [06-cliffwalking-dyna-q](./06-cliffwalking-dyna-q/README.md) | [08-Dyna-Q](../notes/08-Dyna-Q的模型学习与规划更新.md) | `python train.py --episodes 400 --planning-steps 10 --render-final-policy` | `CliffWalking-v1` 上的表格型 `Dyna-Q` |
| [07-cartpole-dqn](./07-cartpole-dqn/README.md) | [09-DQN](../notes/09-DQN的经验回放与目标网络.md) | `python train.py --episodes 400 --print-eval-rollout` | `CartPole-v1` 上的 vanilla `DQN` |
| [08-cartpole-reinforce](./08-cartpole-reinforce/README.md) | [10-REINFORCE](../notes/10-REINFORCE的回合策略梯度与高方差问题.md) | `python train.py --episodes 400` | `CartPole-v1` 上的 `REINFORCE` |
| [09-cartpole-actor-critic](./09-cartpole-actor-critic/README.md) | [11-Actor-Critic](../notes/11-Actor-Critic的价值基线与同步更新.md) | `python train.py --episodes 400` | `CartPole-v1` 上的同步 `Actor-Critic` |
| [10-lunarlander-ppo](./10-lunarlander-ppo/README.md) | [12-PPO](../notes/12-PPO的裁剪目标与稳定策略更新.md) | `python train.py --total-env-steps 200000` | `LunarLander-v3` 上的离散动作 `PPO-Clip` |
| [11-pendulum-sac](./11-pendulum-sac/README.md) | [13-SAC](../notes/13-SAC的最大熵目标与连续动作控制.md) | `python train.py --total-env-steps 10000` | `Pendulum-v1` 上的连续动作 `SAC` |

## 运行前准备

仓库根目录命令如下：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

纯 `pip` 环境建议先安装 `swig`，再安装项目依赖：

```bash
pip install swig
pip install -r requirements.txt
```

更完整的环境说明和命令解释见 [00-环境安装与运行](../notes/00-环境安装与运行.md)。
