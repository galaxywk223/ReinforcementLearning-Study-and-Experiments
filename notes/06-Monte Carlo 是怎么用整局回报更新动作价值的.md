# 蒙特卡洛（Monte Carlo）是怎么用整局回报更新动作价值的

前面的 `Q-Learning` 和 `SARSA` 都属于时序差分（`TD`）思路：

- 走一步
- 立刻更新一步
- 用下一状态当前已有的价值估计来做自举（`bootstrapping`）

这一篇换一个角度：不再立刻更新，而是等一整局结束后，再用整局真实回报去更新。

这就是蒙特卡洛（`Monte Carlo`）方法最核心的区别。

## 先看它和时序差分（TD）最大的不同

`Q-Learning` 和 `SARSA` 在每一步都会构造一个目标值，例如：

$$
r + \gamma \max_{a'} Q(s', a')
$$

或者：

$$
r + \gamma Q(s', a')
$$

它们都没有真的等到整局结束，而是先借用“下一状态当前估计值”继续往后看。

蒙特卡洛（`Monte Carlo`）不这么做。

它会先把整局轨迹记下来：

$$
(s_0, a_0, r_1), (s_1, a_1, r_2), \dots, (s_{T-1}, a_{T-1}, r_T)
$$

等终局出现后，再从后往前算每一步的真实回报：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T
$$

所以它回答的是：

- 这一步之后，这一整局最后到底打成了什么结果

而不是：

- 我先拿当前价值估计拼一个近似目标值

## 为什么 `Blackjack` 很适合学蒙特卡洛（`Monte Carlo`）

因为它天然是：

- 回合制
- 有明确终局
- 每一局很短
- 奖励通常在最后才真正体现

状态可以写成：

$$
(\text{player sum}, \text{dealer showing}, \text{usable ace})
$$

动作只有两个：

- `Stick`
- `Hit`

这使得“先打完整局，再回头给整局里的动作记账”非常直观。

## 固定一个最小例子

先看一局简化后的牌局：

1. 当前状态：`(13, 2, False)`，选择 `Hit`
2. 摸牌后到达：`(20, 2, False)`，即时奖励还是 `0`
3. 然后选择 `Stick`
4. 庄家结算后玩家获胜，最后一步奖励是 `+1`

如果这一篇使用：

$$
\gamma = 1.0
$$

那么这局里两步动作对应的回报分别是：

第二步：

$$
G_1 = 1
$$

第一步：

$$
G_0 = 0 + 1 \times 1 = 1
$$

也就是说，这局里：

- `Hit` 把 `13` 打到 `20`
- `Stick` 把整局收成胜利

这两个动作在这次样本里都会拿到同样的整局回报 `1`。

## 这和 TD 的感觉完全不一样

如果是 `Q-Learning` 或 `SARSA`，第一步更新时不会真的知道这一整局最后能不能赢。

它只能利用：

- 当前一步奖励
- 加上下一状态当前已有的价值估计

但 `Monte Carlo` 会直接等最终结果出来，所以第一步更新时看到的就是：

- 这一步往后走，整局最后真实拿到了多少

因此它没有自举（`bootstrapping`），但代价是：

- 必须等一整局结束
- 不能像时序差分（`TD`）那样边走边学

## 首次访问（`First-Visit`）到底是什么意思

蒙特卡洛（`Monte Carlo`）常见有两种更新方式：

- `First-Visit`
- `Every-Visit`

当前仓库项目实现的是首次访问蒙特卡洛控制（`First-Visit Monte Carlo Control`）。

意思是：

- 对同一局里的某个 `(state, action)`
- 只取它第一次出现时对应的回报来更新

如果这一局里同一个状态动作对后面又出现了，就跳过不再重复记一次。

## 项目里是怎么更新 Q 的

当前项目采用的是“回报样本平均”这一种最直接的实现。

设某个状态动作对已经看过 $N$ 次，当前估计值是：

$$
Q(s, a)
$$

这次又观察到一个整局回报 $G$，则更新成：

$$
Q(s, a) \leftarrow Q(s, a) + \frac{1}{N}\left(G - Q(s, a)\right)
$$

它等价于：

- 把历史上所有该状态动作对对应的回报做平均

所以你可以把蒙特卡洛（`Monte Carlo`）理解成：

- 不断收集完整对局
- 给每个状态动作对累积“赛后总成绩”
- 再用这些真实样本平均来修正价值

## 一个最小数值例子

假设同一个状态动作对：

$$
((13, 2, False), Hit)
$$

连续三次观察到的整局回报分别是：

$$
1,\ -1,\ 1
$$

那么它的 Q 值会依次变成：

第一次：

$$
Q = 1
$$

第二次：

$$
Q = \frac{1 + (-1)}{2} = 0
$$

第三次：

$$
Q = \frac{1 + (-1) + 1}{3} = 0.3333
$$

这就是为什么蒙特卡洛（`Monte Carlo`）的估计经常看起来像：

- 一开始波动很大
- 后面随着样本变多逐渐稳定

## 代码里看哪里

项目代码在这里：

- [train.py](../projects/03-blackjack-monte-carlo/train.py)

最关键的不是单步 `td_target`，而是这两段逻辑：

先在整局结束后反向计算回报：

```python
for index in range(len(episode) - 1, -1, -1):
    _, _, reward = episode[index]
    discounted_return = reward + config.gamma * discounted_return
    returns[index] = discounted_return
```

然后只对这一局第一次出现的状态动作对做样本平均更新：

```python
visit_counts[table_index] += 1
step_size = 1.0 / visit_counts[table_index]
q_table[table_index] += step_size * (returns[index] - q_table[table_index])
```

如果你前面已经习惯了 `Q-Learning / SARSA`，这里最该注意的是：

- 不再有一步一更的 `td_target`
- 更新发生在整局结束之后
- 用的是完整回报，而不是下一状态的当前估计值

## 怎么亲眼看它一步步变

教学脚本见：

- [trace_mc_updates.py](../projects/03-blackjack-monte-carlo/trace_mc_updates.py)

运行：

```bash
cd projects/03-blackjack-monte-carlo
python trace_mc_updates.py --episodes 3
```

这个脚本会打印：

- 每局里每一步的状态、动作和即时奖励
- 整局结束后每一步对应的回报 `G`
- `First-Visit` 样本平均是怎么把 Q 值改掉的

如果你想真正建立蒙特卡洛（`Monte Carlo`）直觉，这个脚本比只盯公式更直接。

## 学完这一篇以后，下一步该补什么

最自然的下一步不是立刻跳深度强化学习，而是把整条价值学习链条补完整：

1. 蒙特卡洛（`Monte Carlo`）
2. `TD(0)`
3. 期望 SARSA（`Expected SARSA`）
4. `n-step TD`
5. 再进入 `DQN`

因为到了这一步，你最需要的不是更多算法名字，而是把：

- 整局回报
- 单步自举
- 多步回报

这三类更新视角真正打通。
