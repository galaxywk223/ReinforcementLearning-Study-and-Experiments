# SARSA和Q-Learning在CliffWalking里会学出什么区别

这篇笔记只看一个问题：同样是表格控制方法，为什么 `SARSA` 和 `Q-Learning` 在 `CliffWalking` 里经常会学出不一样的策略。

## 公式差别

`Q-Learning`：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

`SARSA`：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

关键差别只有一处：

- `Q-Learning` 看下一状态里的最大动作价值
- `SARSA` 看下一状态里真实会执行的动作价值

## 为什么 `CliffWalking` 会放大这个差别

在 `CliffWalking` 里，最短路径通常贴着悬崖边。但训练时策略还带有探索，靠近悬崖意味着走歪一步就可能掉下去。

于是这两个算法会偏向不同的答案：

- `Q-Learning` 更像在问“如果后面都走最好动作，这一步值多少”
- `SARSA` 更像在问“如果后面继续按当前策略走，这一步值多少”

所以常见现象是：

1. `Q-Learning` 的最终路径更短
2. `SARSA` 的最终路径更保守
3. 训练过程中 `SARSA` 的掉崖次数更少

## 看实验时该关注什么

比起只看“最后到没到终点”，更值得看的是：

- 平均回报
- 平均到达步数
- 平均掉崖次数
- 最终贪心策略长什么样

这个环境的关键不在于“能不能到终点”，而在于“到终点的代价和风险有多大”。

## `trace` 和 `compare` 脚本在做不同的事

当前实验目录里有两个脚本：

- [trace_sarsa_updates.py](../experiments/02-cliffwalking-tabular-sarsa/trace_sarsa_updates.py)
- [compare_sarsa_q_learning.py](../experiments/02-cliffwalking-tabular-sarsa/compare_sarsa_q_learning.py)

它们不要混着理解：

- `trace` 脚本固定一条路径，只演示 `SARSA` 的更新过程
- `compare` 脚本让两个算法自己训练，再比较最终结果

所以两边出现不同路径或不同数值是正常的，因为它们本来就在回答不同问题。

## 贪心策略表不是一条实际路径

实验输出里常会打印一张 `greedy policy` 表。那张表表示的是“每个状态下当前最推荐的动作”，而不是已经从起点滚出来的一整条路径。

真正的贪心路径需要从起点出发，一步步按这张表走下去，直到到达终点、掉崖或者进入循环。

## 对应实验

直接运行对比脚本：

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python compare_sarsa_q_learning.py --episodes 800
```

相关内容：

- [04-SARSA是怎么用下一步真实动作更新Q表的](./04-SARSA是怎么用下一步真实动作更新Q表的.md)
- [02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](./03-Q-Learning是怎么一步步把Q表学出来的.md)
