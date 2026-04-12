# SARSA是怎么用下一步真实动作更新Q表的

这篇笔记只看 `SARSA` 和 `Q-Learning` 最关键的一点区别：更新时用的是“下一状态里的最优动作”，还是“下一状态里真实会执行的动作”。

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

所以 $a$ 和 $a'$ 不是同一个动作，而是前后两个时间点的动作。

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

所以 `SARSA` 会把探索时可能走歪的风险一起记到当前动作价值里。

## 为什么用 `CliffWalking`

`CliffWalking-v1` 很适合看这个差别，因为最短路通常贴着悬崖边，而训练时一旦探索出错，就可能掉下悬崖并回到起点。

在这种环境里：

- `Q-Learning` 更容易偏向理论上更短的路
- `SARSA` 更容易学到离悬崖远一点的安全路径

## 固定一条安全路径看更新

为了看清数值传播，可以先固定一条不踩悬崖的路径：

```text
U -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> D
```

这条路径总共 `13` 步，每一步奖励都是 `-1`。

第一轮训练时，路径上的动作通常都会先变成负值，因为即时奖励本身就是负的。第二轮再走同一条路径时，前面的动作会开始感受到“后面还有很多步代价”，于是更早的状态也会继续变小。

这和 `Q-Learning` 一样都有“价值传播”，不同的是 `SARSA` 传播的是“按当前策略真实会发生的后续代价”。

## 代码位置

训练脚本：

- [train.py](../experiments/02-cliffwalking-tabular-sarsa/train.py)

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

## 教学脚本

- [trace_sarsa_updates.py](../experiments/02-cliffwalking-tabular-sarsa/trace_sarsa_updates.py)

运行：

```bash
cd experiments/02-cliffwalking-tabular-sarsa
python trace_sarsa_updates.py --episodes 2
```

## 对应内容

- [05-SARSA和Q-Learning在CliffWalking里会学出什么区别](./05-SARSA和Q-Learning在CliffWalking里会学出什么区别.md)
- [02-cliffwalking-tabular-sarsa](../experiments/02-cliffwalking-tabular-sarsa/README.md)
- [03-Q-Learning是怎么一步步把Q表学出来的](./03-Q-Learning是怎么一步步把Q表学出来的.md)
