# 学习笔记

这个目录收录仓库的公开学习笔记，文件名数字表示推荐阅读顺序。

## 笔记列表

- [00-环境安装](./00-环境安装.md)
- [01-第一次理解强化学习](./01-第一次理解强化学习.md)
- [02-马尔可夫决策过程（MDP）、回报与 Bellman 方程](./02-MDP、回报与 Bellman 方程.md)
- [03-Q 学习（Q-Learning）是怎么一步步把 Q 表学出来的](./03-Q-Learning 是怎么一步步把 Q 表学出来的.md)
- [04-SARSA 是怎么用“下一步真实动作”更新 Q 表的](./04-SARSA 是怎么用“下一步真实动作”更新 Q 表的.md)
- [05-SARSA 和 Q 学习（Q-Learning）在 CliffWalking 里会学出什么区别](./05-SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别.md)
- [06-蒙特卡洛（Monte Carlo）是怎么用整局回报更新动作价值的](./06-Monte Carlo 是怎么用整局回报更新动作价值的.md)

## 建议阅读顺序

目录和文件名前缀数字表示推荐阅读顺序。

1. 先读 `01-第一次理解强化学习.md`
2. 再读 `02-MDP、回报与 Bellman 方程.md`，建立对马尔可夫决策过程和 Bellman 递推的基本直觉
3. 配合 `projects/01-frozenlake-tabular-q/` 阅读 `03-Q-Learning 是怎么一步步把 Q 表学出来的.md`
4. 再配合 `projects/02-cliffwalking-tabular-sarsa/` 阅读 `04-SARSA 是怎么用“下一步真实动作”更新 Q 表的.md`
5. 最后阅读 `05-SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别.md`，对比两种表格控制方法
6. 然后配合 `projects/03-blackjack-monte-carlo/` 阅读 `06-Monte Carlo 是怎么用整局回报更新动作价值的.md`，理解整局回报和首次访问（`First-Visit`）更新
