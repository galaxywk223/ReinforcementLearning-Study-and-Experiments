# 实验代码

这个目录只负责代码和运行入口。建议先从 [notes/README.md](../notes/README.md) 开始阅读，再按需要进入具体实验目录运行脚本。

## 当前实验

| 目录 | 主笔记 | 主入口 | 说明 |
| --- | --- | --- | --- |
| [01-frozenlake-tabular-q](./01-frozenlake-tabular-q/README.md) | [03-Q-Learning](../notes/03-Q-Learning的值传播与Q表更新.md) | `python train.py --episodes 4000 --render-final-policy` | `FrozenLake-v1` 上的表格型 `Q-Learning` |
| [02-cliffwalking-tabular-sarsa](./02-cliffwalking-tabular-sarsa/README.md) | [04-SARSA](../notes/04-SARSA的时序更新与策略差异.md) | `python train.py --episodes 800 --render-final-policy` | `CliffWalking-v1` 上的表格型 `SARSA` |
| [03-blackjack-monte-carlo](./03-blackjack-monte-carlo/README.md) | [05-MonteCarlo](../notes/05-MonteCarlo的整局回报与动作价值更新.md) | `python train.py --episodes 200000 --render-final-policy` | `Blackjack-v1` 上的首次访问蒙特卡洛控制 |

## 运行前准备

推荐在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

或者：

```bash
pip install -r requirements.txt
```

更完整的概念解释和结果解读，都已经回到对应主笔记里。
