# SARSA是怎么用下一步真实动作更新Q表的

这篇笔记把 `SARSA` 的公式、`CliffWalking` 里的策略差异，以及代表实验观察放到一起看。核心问题只有一个：为什么把“下一步真实会选到的动作”带进更新之后，学出来的策略会更保守。

## `SARSA` 更新式

`Q-Learning` 的目标值是：

$$
r + \gamma \max_{a'} Q(s', a')
$$

`SARSA` 的目标值是：

$$
r + \gamma Q(s', a')
$$

差别只在最后这一项，但含义很不一样：

- `Q-Learning` 默认后面总能选到最好的动作
- `SARSA` 会把下一步真实选到的动作一起算进去

完整更新式：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

## 为什么名字里有两个 `A`

这里的时序是：

$$
(s, a, r, s', a')
$$

也就是：

1. 当前在状态 $s$
2. 执行动作 $a$
3. 得到奖励 $r$
4. 转移到下一状态 $s'$
5. 再从 $s'$ 里选出下一步动作 $a'$

所以 `SARSA` 不是只看下一状态，而是把“下一状态 + 下一动作”一起写进目标值。

## 一个最小例子

假设当前奖励 $r = -1$，折扣因子 $\gamma = 0.9$。到了下一状态 $s'$ 后，当前 Q 值是：

- `Up = 10`
- `Right = 7`
- `Down = 3`
- `Left = -5`

那么：

- `Q-Learning` 一定会用最大值 `10`
- `SARSA` 要看下一步实际选到哪个动作

如果 `SARSA` 下一步选到 `Left`，目标值就是：

$$
-1 + 0.9 \times (-5) = -5.5
$$

如果下一步选到 `Up`，目标值才会变成：

$$
-1 + 0.9 \times 10 = 8
$$

这就是 `SARSA` 的直觉核心：它会把探索时可能走歪的风险一起记进当前动作价值。

## 为什么用 `CliffWalking`

`CliffWalking-v1` 很适合看这个差别，因为最短路通常贴着悬崖边，而训练时策略还带有探索，一旦走歪就可能掉下悬崖并回到起点。

于是这两个算法会自然分开：

- `Q-Learning` 更像在问：“如果后面都选最好动作，这一步值多少？”
- `SARSA` 更像在问：“如果后面继续按当前策略走，这一步值多少？”

所以常见现象是：

- `Q-Learning` 更容易偏向更短路径
- `SARSA` 更容易学到离悬崖远一点的安全路径

## 固定一条安全路径看更新

为了把数值传播看清楚，可以先固定一条不踩悬崖的路径：

```text
U -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> D
```

这条路径总共 `13` 步，每一步奖励都是 `-1`。

第一轮训练时，路径上的动作通常都会先变成负值，因为即时奖励本身就是负的。第二轮再走同一条路径时，前面的动作会开始感受到“后面还有很多步代价”，于是更早的状态也会继续变小。

这和 `Q-Learning` 一样都有“价值传播”，不同的是 `SARSA` 传播的是“按当前策略真实会发生的后续代价”。

## 放到完整训练里会看到什么

当前仓库的 `SARSA` 基线实验结果是：

- 回合数：`800`
- 评估平均回报：`-17.0`
- 平均到达步数：`17.0`
- 平均掉崖次数：`0.0`

![CliffWalking reward curve](../assets/figures/cliffwalking/reward_curve.png)

这里最值得看的不是“是否到终点”，而是：

- 它虽然不是最短路，但稳定到达终点
- 评估阶段几乎不掉崖
- 这正说明 `SARSA` 把探索风险算进了当前动作价值

也因此，实验里的 `greedy policy` 表不能直接当成“一条已经走出来的路径”。那张表表示的是“每个状态最推荐的动作”；真正路径还要从起点一步步滚出来，才知道会不会掉崖、会不会进入循环。

## 和 `Q-Learning` 对比时该看什么

如果把 `SARSA` 和 `Q-Learning` 放到同一个 `CliffWalking` 里比较，建议重点看：

- 平均回报
- 平均到达步数
- 平均掉崖次数
- 最终贪心策略是不是贴着悬崖边

这个环境的关键不在于“能不能到终点”，而在于“到终点的代价和风险有多大”。

## 代码位置

训练脚本：

- [train.py](../experiments/02-cliffwalking-tabular-sarsa/train.py)

直接运行：

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python train.py --episodes 800 --render-final-policy
python compare_sarsa_q_learning.py --episodes 800
```

核心更新：

```python
next_action = epsilon_greedy_action(q_table, next_state, epsilon, env.action_space, rng)
td_target = reward + config.gamma * q_table[next_state, next_action]
td_error = td_target - q_table[state, action]
q_table[state, action] += config.alpha * td_error
```

最关键的是：

```python
q_table[next_state, next_action]
```

这里明确把“下一状态 + 真实下一动作”一起带入了更新。

## 教学脚本和对比脚本

- [trace_sarsa_updates.py](../experiments/02-cliffwalking-tabular-sarsa/trace_sarsa_updates.py)
- [compare_sarsa_q_learning.py](../experiments/02-cliffwalking-tabular-sarsa/compare_sarsa_q_learning.py)

两者回答的问题不一样：

- `trace` 脚本固定一条路径，只演示 `SARSA` 的更新过程
- `compare` 脚本让两个算法自己训练，再比较最终结果

## 对应内容

- [02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](./03-Q-Learning是怎么一步步把Q表学出来的.md)
- [05-MonteCarlo是怎么用整局回报更新动作价值的](./05-MonteCarlo是怎么用整局回报更新动作价值的.md)
