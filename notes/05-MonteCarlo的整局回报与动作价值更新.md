# MonteCarlo的整局回报与动作价值更新

前面的 `Q-Learning` 和 `SARSA` 都是在走一步之后就立刻更新。这篇换一个角度：先把一整局打完，再回头用整局真实回报更新动作价值。

## 和时序差分最大的不同

`Q-Learning` 和 `SARSA` 都会在每一步构造一个目标值，例如：

$$
r + \gamma \max_{a'} Q(s', a')
$$

或者：

$$
r + \gamma Q(s', a')
$$

它们都借用了“下一状态当前的价值估计”，也就是常说的自举。

`Monte Carlo` 不这样做。它会先把整局轨迹记下来：

$$
(s_0, a_0, r_1), (s_1, a_1, r_2), \dots, (s_{T-1}, a_{T-1}, r_T)
$$

等终局出现后，再从后往前计算每一步的整局回报：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

## 为什么用 `Blackjack`

`Blackjack-v1` 便于观察 `Monte Carlo`，因为它：

- 是回合制环境
- 每局都一定结束
- 状态和动作空间都比较清楚
- 很多奖励要到回合结束后才真正体现

状态可以写成：

$$
(\text{player sum}, \text{dealer showing}, \text{usable ace})
$$

动作只有两个：`Stick` 和 `Hit`。

## 一个最小例子

假设有一局牌：

1. 当前状态是 `(13, 2, False)`，选择 `Hit`
2. 摸牌后到达 `(20, 2, False)`，即时奖励还是 `0`
3. 然后选择 `Stick`
4. 结算后玩家获胜，最后一步奖励是 `+1`

如果 $\gamma = 1.0$，那么这局里两步动作的整局回报都是 `1`。也就是说，第一步虽然当下没有奖励，但它最终还是因为这局赢了而拿到正回报。

这说明 `Monte Carlo` 的动作值不是只看眼前奖励，而是依赖整局最终结果。

## `First-Visit` 是什么意思

当前仓库实现的是 `First-Visit Monte Carlo Control`。

意思是：

- 对同一局里的某个 `(state, action)`
- 只用它第一次出现时对应的整局回报来更新

如果同一局后面又出现了同一个状态动作对，就跳过。

## 更新是怎么做的

项目里使用的是回报样本平均。设某个状态动作对已经看过 $N$ 次，这次观察到整局回报 $G$，则：

$$
Q(s, a) \leftarrow Q(s, a) + \frac{1}{N}\left(G - Q(s, a)\right)
$$

可以把它理解成：不断收集完整对局，然后对这个状态动作对历史上看到的整局结果做平均。

## 放到完整训练里会看到什么

当前仓库的 `Blackjack` 基线实验结果是：

- 回合数：`500000`
- 评估平均回报：`-0.0413`
- 胜率：`0.4350`
- 平局率：`0.0887`
- 负率：`0.4763`

![Blackjack policy heatmaps](../assets/figures/blackjack/policy_heatmaps.png)

这个实验最值得看的不是“平均回报是不是很高”，而是策略边界会逐渐变清楚：

- 没有可用 `A` 时，玩家通常在更保守的点数停牌
- 有可用 `A` 时，策略会更敢继续要牌
- 同一个玩家点数下，庄家明牌不同会改变最优动作

这些现象可以直接用 `Monte Carlo` 的更新方式来解释，因为它们本来就依赖“整局最后到底赢没赢”的真实结果。

如果想看训练曲线，也可以再配合：

![Blackjack reward curve](../assets/figures/blackjack/reward_curve.png)

## 代码位置

训练脚本：

- [train.py](../experiments/03-blackjack-monte-carlo/train.py)

直接运行：

```bash
cd experiments/03-blackjack-monte-carlo
python train.py --episodes 200000 --render-final-policy
```

先在回合结束后反向计算回报：

```python
for index in range(len(episode) - 1, -1, -1):
    _, _, reward = episode[index]
    discounted_return = reward + config.gamma * discounted_return
    returns[index] = discounted_return
```

再对首次访问到的状态动作对做样本平均更新：

```python
visit_counts[table_index] += 1
step_size = 1.0 / visit_counts[table_index]
q_table[table_index] += step_size * (returns[index] - q_table[table_index])
```

## 回报追踪脚本

- [trace_mc_updates.py](../experiments/03-blackjack-monte-carlo/trace_mc_updates.py)

运行：

```bash
cd experiments/03-blackjack-monte-carlo
python trace_mc_updates.py --episodes 3
```

这个脚本会把每局里的状态、动作、即时奖励和最终回报一起打印出来，便于把整局回报和状态动作更新直接对应起来。

## 对应内容

- [03-blackjack-monte-carlo](../experiments/03-blackjack-monte-carlo/README.md)
- [04-SARSA的时序更新与策略差异](./04-SARSA的时序更新与策略差异.md)
