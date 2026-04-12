# SARSA 和 Q 学习（Q-Learning）在 CliffWalking 里会学出什么区别

当你已经学过：

- `Q-Learning`
- `Bellman` 更新
- $\epsilon$-greedy

下一步最值得弄清楚的问题通常不是“再多背一个算法名字”，而是：

- `SARSA` 和 `Q-Learning` 到底差在哪
- 这种差别什么时候会真正表现出来

`CliffWalking` 正是最经典的观察环境之一。

## 先看公式差别

`Q-Learning`：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

`SARSA`：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

唯一关键差别是：

- `Q-Learning` 用下一状态的最大动作价值
- `SARSA` 用下一状态真实会采取的那个动作价值

因此：

- `Q-Learning` 是离策略学习（`off-policy`）
- `SARSA` 是按当前策略学习（`on-policy`）

## CliffWalking 为什么特别适合看这个差别

在 `CliffWalking` 中：

- 从起点到终点的最短路，通常沿着悬崖边
- 但靠近悬崖意味着，只要探索时走歪一步，就可能踩空

这正好把两种算法的性格放大出来。

### Q 学习（Q-Learning）更像什么

它更新时想的是：

- 下一状态里最好的动作是什么

所以它容易学出：

- 靠近悬崖的最短路径

因为在它看来，只要后续都做最优动作，那条路的理论回报更高。

### SARSA 更像什么

它更新时想的是：

- 下一步按我当前这套带探索的策略，真实会怎么走

所以它更容易学出：

- 离悬崖远一点的安全路径

因为它把训练期可能发生的误操作风险也算进来了。

## 不是“谁更高级”，而是“谁在回答不同的问题”

很多初学者会问：

- 到底哪个更好

更准确的理解是：

- `Q-Learning` 更接近“如果后面都做最好动作，这一步值多少”
- `SARSA` 更接近“如果后面继续按当前策略走，这一步值多少”

所以在带风险、带探索的训练期里，`SARSA` 往往显得更稳。

## 在 CliffWalking 里通常会看到什么现象

一个常见现象是：

1. `Q-Learning` 的最终策略更短
2. `SARSA` 的最终策略更绕
3. 但训练过程中，`SARSA` 的悬崖掉落次数更少

也就是说：

- `Q-Learning` 经常追求更优的理论路径
- `SARSA` 经常选择更安全的实际路径

## 最值得观察的不是单次成败，而是这些指标

在 `CliffWalking` 里，比起只看“最后到没到终点”，更应该看：

- 平均回报
- 平均到达步数
- 平均掉下悬崖次数
- 学出来的最终策略长什么样

因为这个环境里：

- 很多策略最终都能到终点
- 但沿路承受的风险和惩罚可能差很多

## 当前仓库里怎么复现

当前仓库给了一个直接对比脚本：

- [compare_sarsa_q_learning.py](../projects/02-cliffwalking-tabular-sarsa/compare_sarsa_q_learning.py)

运行：

```bash
cd projects/02-cliffwalking-tabular-sarsa
python compare_sarsa_q_learning.py --episodes 800
```

它会输出并保存：

- `SARSA` 的训练曲线
- `Q-Learning` 的训练曲线
- 两种算法的评估结果摘要
- 两种最终贪心策略

## 看结果时怎么解读

如果你看到：

- `Q-Learning` 路径更贴近悬崖
- `SARSA` 路径更远离悬崖

这通常不是训练失败，而正是这两个算法的典型差异。

因为它们更新时回答的是两类不同问题：

- 最优控制角度的价值估计
- 当前策略角度的价值估计

## 为什么 `trace` 脚本和 `compare` 脚本里的 SARSA 不会长一样

当前仓库里有两个很容易混淆的脚本：

- [trace_sarsa_updates.py](../projects/02-cliffwalking-tabular-sarsa/trace_sarsa_updates.py)
- [compare_sarsa_q_learning.py](../projects/02-cliffwalking-tabular-sarsa/compare_sarsa_q_learning.py)

它们的任务完全不同。

`trace_sarsa_updates.py` 做的是：

- 固定一条人为指定的安全路径
- 不让算法自己决定怎么走
- 只演示 `SARSA` 更新时为什么要用 $Q(s', a')$

所以它回答的是：

- 在一条已知路径上，数值会怎么传播

而 `compare_sarsa_q_learning.py` 做的是：

- 让 `SARSA` 和 `Q-Learning` 真正在环境里训练
- 每一步都按 $\epsilon$-greedy 自己选动作
- 最后再比较它们学出来的策略

所以它回答的是：

- 真实训练后，这两种算法会学成什么样

因此：

- `trace` 里的路径是人为固定的
- `compare` 里的路径是训练出来的

两边结果不同不是出错，而正是因为它们在做两件不同的事。

## 贪心策略（`greedy policy`）表和“实际贪心路径”也不是一回事

实验输出里常会打印一张 `greedy policy` 表。

它的意思是：

- 对每一个状态 $s$
- 单独取当前 Q 值最大的动作

也就是：

$$
\pi(s) = \arg\max_a Q(s, a)
$$

这张表本质上是在回答：

- 如果你来到这个状态，我会建议你走哪个动作

但这还不是“一条实际路径”。

真正的贪心路径应该这样读：

1. 从起点 `S` 开始
2. 看起点对应的贪心动作
3. 走到下一个状态
4. 再看新状态对应的贪心动作
5. 一直重复，直到到达终点、掉下悬崖，或者进入循环

所以：

- `greedy policy` 是整张状态到动作的映射表
- `greedy path` 是从起点把这些局部动作串起来后的实际轨迹（`rollout`）

这也是为什么：

- 一张策略表看起来很复杂
- 但从起点真正走出来的实际路径通常只是一条线

## 评估时为什么和训练时也可能不一样

在训练阶段，脚本使用的是 $\epsilon$-greedy：

- 大部分时候按当前 Q 表选动作
- 少部分时候随机探索

但在评估阶段，脚本使用的是纯贪心策略：

- 永远选 $\arg\max_a Q(s, a)$
- 不再额外探索

所以训练期和评估期回答的问题也不一样：

- 训练期：带探索时整体表现如何
- 评估期：学出来的贪心策略本身表现如何

这也是为什么在 `CliffWalking` 中：

- `SARSA` 训练时会更在意探索风险
- 但最后打印出来的评估路径仍然是纯贪心路径

## 建议把这几个问题带着去看实验

跑实验时，重点不要只盯平均值，还要问：

1. 最终策略是不是沿悬崖边走
2. 训练期间哪种算法掉崖更多
3. 当 $\epsilon$ 变大时，哪种算法更受影响
4. 当 $\epsilon$ 固定不衰减时，哪种算法更容易学得保守

这些问题会直接帮你把：

- 按当前策略学习（`on-policy`）
- `off-policy`
- 训练期风险
- 最终贪心策略

几件事连起来。

## 配合哪些内容一起看

- [SARSA 是怎么用“下一步真实动作”更新 Q 表的](./04-SARSA 是怎么用“下一步真实动作”更新 Q 表的.md)
- [Q 学习（Q-Learning）是怎么一步步把 Q 表学出来的](./03-Q-Learning 是怎么一步步把 Q 表学出来的.md)
- [CliffWalking 表格型 SARSA 项目说明](../projects/02-cliffwalking-tabular-sarsa/README.md)
