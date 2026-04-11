# SARSA 是怎么用“下一步真实动作”更新 Q 表的

这一篇继续停留在表格方法，但把重点从 `Q-Learning` 换到 `SARSA`：

1. 它更新时到底多看了什么
2. 为什么它常被说成更“保守”
3. 为什么在 `CliffWalking` 里它经常学出更安全的路径

## SARSA 更新的核心区别

`Q-Learning` 的目标值是：

$$
r + \gamma \max_{a'} Q(s', a')
$$

它会直接看“下一状态里最好的动作价值”。

而 `SARSA` 的目标值是：

$$
r + \gamma Q(s', a')
$$

这里的 $a'$ 不是“最大值动作”，而是：

- 智能体在下一状态
- 按当前策略
- 真实会选到的那个动作

完整更新公式是：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

所以 `SARSA` 不是在问：

- 下一步理论上最好的动作值是多少

而是在问：

- 下一步按我现在这套带探索的策略，实际很可能会怎么走

这就是它常被叫做 `on-policy` 的原因。

## `a` 和 `a'` 不是同一个动作

很多人第一次看 `SARSA` 时，会疑惑：

- 当前动作已经记成 `a`
- 为什么下一项里又冒出一个 `a'`
- 它是不是还是原来那个动作

不是。

这里的 `a` 和 `a'` 分别属于两个相邻时间点：

1. 当前在状态 `s`
2. 先执行当前动作 `a`
3. 环境返回奖励 `r`
4. 环境转移到下一状态 `s'`
5. 到了 `s'` 后，再选下一步动作 `a'`

所以：

- `a` 是“这一步”的动作
- `a'` 是“下一步”的动作

把这串时序连起来看，就是：

$$
(s, a, r, s', a')
$$

这也是 `SARSA` 这个名字的来源：

- `State`
- `Action`
- `Reward`
- `State`
- `Action`

## `r + \gamma Q(s', a')` 和 `r + \gamma \max_{a'} Q(s', a')` 到底差在哪

两者都先看这一步拿到的即时奖励 `r`，真正不同的是：

- 下一状态 `s'` 的未来价值，到底按“真实下一动作”算
- 还是按“理论上最好的下一动作”算

`SARSA`：

$$
r + \gamma Q(s', a')
$$

意思是：

- 我已经到了 `s'`
- 接下来按当前策略真实会选到某个动作 `a'`
- 就用这个动作的价值继续往后算

`Q-Learning`：

$$
r + \gamma \max_{a'} Q(s', a')
$$

意思是：

- 我已经到了 `s'`
- 不管下一步真实会不会探索走歪
- 我都直接假设后面能选到最好的动作

所以它们回答的是两类不同问题：

- `SARSA`：如果后面继续按当前策略走，这一步值多少
- `Q-Learning`：如果后面都按最优动作走，这一步值多少

## 一个最小数值例子

假设：

- 当前奖励 `r = -1`
- 折扣因子 `\gamma = 0.9`

并且到了下一状态 `s'` 后，四个动作的当前 Q 值分别是：

- `Up = 10`
- `Right = 7`
- `Down = 3`
- `Left = -5`

那么：

- `Q-Learning` 一定取最大值 `10`
- `SARSA` 要看下一步真实选到的是哪个动作

于是：

`Q-Learning` 的目标值是：

$$
-1 + 0.9 \times 10 = 8
$$

如果 `SARSA` 下一步真实选到的是 `Left`，目标值就是：

$$
-1 + 0.9 \times (-5) = -5.5
$$

如果 `SARSA` 下一步真实选到的是 `Up`，目标值才会变成：

$$
-1 + 0.9 \times 10 = 8
$$

所以关键不是公式表面上只差一个 `max`，而是：

- `Q-Learning` 默认后面总能走最好
- `SARSA` 会把下一步真实可能发生的动作也算进去

## 先固定一个实验：CliffWalking

这一篇使用：

- 环境：`CliffWalking-v1`
- 算法：`Tabular SARSA`

环境可以想成一个 `4 x 12` 网格：

```text
Row 0: . . . . . . . . . . . .
Row 1: . . . . . . . . . . . .
Row 2: . . . . . . . . . . . .
Row 3: S C C C C C C C C C C G
```

其中：

- `S`：起点
- `G`：终点
- `C`：悬崖

如果踩到悬崖：

- 会拿到大额负奖励
- 并被送回起点

因此它很适合观察“算法到底会不会主动远离风险”。

## 我们更新的是什么

还是 Q 表里的一个元素：

$$
Q(s, a)
$$

表示：

在状态 $s$ 下执行动作 $a$ 的长期价值估计。

和 `Q-Learning` 一样，`SARSA` 也是逐步修正这个估计；不同之处只是目标值里用的是：

$$
Q(s', a')
$$

而不是：

$$
\max_{a'} Q(s', a')
$$

## 一个最小局部例子：同一个下一状态，不同下一动作

假设当前从状态 $s$ 走到下一状态 $s'$，这一跳拿到的奖励是：

$$
r = -1
$$

此时如果在 $s'$ 下：

- 真实选到安全动作 `Up`
- 并且 $Q(s', \text{Up}) = -1$

那么 `SARSA` 的目标值是：

$$
-1 + \gamma \cdot (-1)
$$

但如果在同一个 $s'$ 下：

- 因为探索选到了更危险的动作 `Right`
- 并且 $Q(s', \text{Right}) = -5$

那么目标值就会变成：

$$
-1 + \gamma \cdot (-5)
$$

这说明 `SARSA` 会把“下一步真实可能做出的危险动作”一起算进去。

这正是它在悬崖环境里更保守的根源。

## 固定一条安全路径看更新

为了把更新过程看清楚，可以先固定一条不踩悬崖的路径：

```text
U -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> R -> D
```

也就是：

- 先向上离开悬崖边
- 再一路向右
- 最后向下走到终点

这一条路径总共 `13` 步，而且每一步奖励都是 `-1`。

## 第 1 轮：路径上的动作先都变成负值

训练开始前，Q 表是全零。

在第一轮固定路径中，每一步更新时：

- 当前奖励都是 `-1`
- 下一动作对应的 Q 值还是 `0`

所以目标值都是：

$$
-1 + \gamma \cdot 0 = -1
$$

如果取：

$$
\alpha = 0.5
$$

那么路径上的动作在第一轮后都会更新成：

$$
-0.5
$$

这说明：即使是一条安全路径，只要每一步都有代价，它的 Q 值也会先学成负数。

## 第 2 轮：路径代价开始往前传播

第二轮再走同一条路径时，情况就不同了。

例如起点 `S` 向上走的这一步，更新时会看到：

- 当前奖励还是 `-1`
- 下一状态中，真实下一动作是 `Right`
- 而上一轮已经学到这个动作的价值大约是 `-0.5`

于是目标值变成：

$$
-1 + 0.99 \cdot (-0.5) = -1.495
$$

如果旧值还是 `-0.5`，则：

$$
Q(s, a) \leftarrow -0.5 + 0.5 \cdot (-1.495 + 0.5) = -0.9975
$$

这说明：

- 不只是当前这一步有代价
- 后面的路径长度代价也开始一轮一轮向前传播

## 和 Q-Learning 的真正区别在哪

只看公式时，很多人会觉得两者差别很小。

但在 `CliffWalking` 里，差别非常具体：

- `Q-Learning` 更新时看的是下一状态最好的动作值
- `SARSA` 更新时看的是下一状态真实会执行的动作值

如果当前策略里还保留探索，那么 `SARSA` 会把“下一步可能误踩悬崖”的风险也计入当前动作价值。

因此它更容易学到：

- 离悬崖远一点
- 虽然步数更多
- 但训练期整体更稳的策略

## 对应代码看哪里

当前仓库里的 `SARSA` 项目在这里：

- [train.py](../../projects/cliffwalking-tabular-sarsa/train.py)

核心更新逻辑是这几行：

```python
next_action = epsilon_greedy_action(q_table, next_state, epsilon, env.action_space, rng)
td_target = reward + config.gamma * q_table[next_state, next_action]
td_error = td_target - q_table[state, action]
q_table[state, action] += config.alpha * td_error
```

最应该盯住的是：

```python
q_table[next_state, next_action]
```

因为这里体现的就是：

- 下一状态
- 配合真实下一动作
- 一起进入本次更新

## 怎么亲眼看它一步步变

教学脚本见：

- [trace_sarsa_updates.py](../../projects/cliffwalking-tabular-sarsa/trace_sarsa_updates.py)

运行：

```bash
cd projects/cliffwalking-tabular-sarsa
python trace_sarsa_updates.py --episodes 2
```

这个脚本会固定沿安全路径前进，并打印：

- 当前状态
- 当前动作
- 下一状态
- 真实下一动作
- 当前奖励
- 下一动作对应的 Q 值
- 目标值
- 更新前 Q 值
- 更新后 Q 值

如果你已经理解了 `Q-Learning`，这一篇最重要的任务就是看懂：

- `SARSA` 的更新不再问“理论最优”
- 而是问“按当前策略，下一步真实怎么走”

## 接下来读什么

- [Q-Learning 是怎么一步步把 Q 表学出来的](./q-learning-step-by-step.md)
- [SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别](./sarsa-vs-q-learning.md)
- [CliffWalking Tabular SARSA 项目说明](../../projects/cliffwalking-tabular-sarsa/README.md)
