# DQN的经验回放与目标网络

本节讨论 `DQN` 如何把 `Q-Learning` 从表格型状态推广到连续状态。此前章节里的 `FrozenLake`、`CliffWalking` 和 `Blackjack` 都可以直接维护 `Q(s, a)` 表；`CartPole-v1` 的状态是连续实数向量，不能再把每个状态逐个列出来，因此需要用神经网络近似动作价值函数。

## 为什么表格法在 `CartPole` 上不再适合

`CartPole` 的单个状态由四个连续量组成：

$$
s = (x, \dot{x}, \theta, \dot{\theta})
$$

其中小车位置、速度、杆角度和角速度都不是离散编号。对于这种状态空间，如果继续沿用：

$$
Q(s, a)
$$

就必须先把连续状态强行离散化，或者维护一个几乎无穷大的查找表。`DQN` 的核心改动就是把表格改成参数化函数：

$$
Q_\theta(s, a)
$$

这里的参数 $\theta$ 由神经网络学习。

## `DQN` 的目标值仍然来自 `Q-Learning`

`DQN` 没有改变 `Q-Learning` 的基本目标，只是把“查表”换成了“前向计算”。对一条转移：

$$
(s_t, a_t, r_{t+1}, s_{t+1})
$$

它的目标值仍然是：

$$
y_t =
\begin{cases}
r_{t+1}, & \text{如果到达终止状态} \\
r_{t+1} + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a'), & \text{否则}
\end{cases}
$$

其中：

- `online network` 负责产生当前预测值 `Q_\theta(s_t, a_t)`
- `target network` 负责产生更稳定的目标值 `Q_{\theta^-}(s_{t+1}, a')`

损失函数写成：

$$
L(\theta) = \mathbb{E}\left[\ell\left(Q_\theta(s_t, a_t), y_t\right)\right]
$$

当前仓库实现使用 `Huber loss`，因为它在误差较大时比纯平方误差更稳。

## 为什么需要经验回放

如果每次都直接用“刚刚发生的那一步”训练网络，会遇到两个问题：

- 连续样本高度相关，梯度方向抖动明显
- 新样本会立刻覆盖旧经验，训练目标变化过快

经验回放的做法如下：

- 先把转移 `(state, action, reward, next_state, done)` 存进 `replay buffer`
- 每次更新时随机抽一批样本
- 用 minibatch 一次性构造多个 `TD target`

这样做的作用是把相邻轨迹打散，让一次更新看到更混合的经验分布。

## 为什么还要目标网络

如果同一个网络既负责预测当前值，又负责生成下一状态的目标值，那么目标值会随着每次参数更新一起漂移。`DQN` 因此引入目标网络：

- `policy_net` 每一步持续被优化
- `target_net` 每隔若干次优化才从 `policy_net` 拷贝一次参数

当前仓库的默认做法是每 `200` 次优化进行一次硬同步。这样目标值不会在每一次反向传播时同步变化，训练会更稳定。

## 一个最小批量更新例子

假设 replay buffer 里随机抽到一条样本：

$$
s = [0.02, 0.10, 0.03, 0.15], \quad a = 1, \quad r = 1
$$

下一状态为：

$$
s' = [0.022, 0.29, 0.033, -0.12]
$$

如果目标网络在 `s'` 上输出两个动作值：

$$
Q_{\theta^-}(s', \cdot) = [0.84, 0.91]
$$

那么当 $\gamma = 0.99$ 时：

$$
y = 1 + 0.99 \times 0.91 = 1.9009
$$

如果当前网络对被选动作的预测值是：

$$
Q_\theta(s, a=1) = 1.42
$$

那么这条样本的更新方向就是把 `1.42` 往 `1.9009` 推近。

如果样本本身已经终止，则自举项消失，只保留：

$$
y = r
$$

这也是 `done=True` 时目标值与普通样本的关键差异。

## 为什么用 `CartPole`

`CartPole-v1` 适合作为第一章 `DQN` 实验，原因如下：

- 动作空间离散，仍然可以直接输出每个动作的 `Q` 值
- 奖励规则简单，每坚持一步就得到 `+1`
- 训练速度快，CPU 即可跑通
- 现象足够清楚，适合从表格法过渡到神经网络近似

它并不展示稀疏奖励或高维观测，而是专门用于把“值函数表”升级成“值函数网络”。

## 当前仓库实现了什么

当前章节对应实验采用最小 vanilla `DQN` 配置：

- 环境：`CartPole-v1`
- 网络：`4 -> 128 -> 128 -> 2`
- 优化器：`Adam`
- 损失：`Huber loss`
- 回放池容量：`20000`
- batch size：`64`
- `learning_starts = 1000`
- 目标网络同步间隔：`200` 次优化

训练流程如下：

1. 用 `epsilon-greedy` 与环境交互，收集转移
2. 把转移放入 `replay buffer`
3. 达到 `learning_starts` 后，每一步随机采样一个 minibatch
4. 用 `target_net` 构造 `TD target`
5. 用 `policy_net` 计算被选动作的预测值并反向传播
6. 每隔固定优化步数同步一次 `target_net`

## 放到完整训练里会看到什么

当前仓库的 `CartPole` 基线实验结果是：

- 回合数：`400`
- 评估平均回报：`500.0`
- 评估平均回合长度：`500.0`
- 成功率：`1.0`

<p align="center">
  <img src="../assets/figures/cartpole-dqn/reward_curve.png" alt="CartPole DQN 奖励曲线" width="920" />
</p>

完整训练时最典型的现象有两类：

- 训练前期奖励接近随机水平，因为经验池还在累积、探索率也很高
- 训练中后期奖励曲线和评估回合长度明显抬升，说明网络已经学会维持杆平衡更久

与表格型 `Q-Learning` 相比，这一章的重点不再是“某个格子的值如何沿路径传播”，而是：

- 神经网络如何对未见过的连续状态做近似
- 回放采样如何打散样本相关性
- 目标网络如何减缓训练目标漂移

## 代码位置

训练脚本：

- [train.py](../experiments/05-cartpole-dqn/train.py)

直接运行：

```bash
cd experiments/05-cartpole-dqn
python train.py --episodes 400 --print-eval-rollout
```

核心目标值构造如下：

```python
predicted_q = policy_net(states).gather(1, actions).squeeze(1)
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    td_target = rewards + gamma * next_q * (1.0 - dones)
```

这段代码对应的含义如下：

- `policy_net(states).gather(1, actions)` 只取出 minibatch 中“实际执行过的动作”的预测值
- `target_net(next_states).max(dim=1)` 为下一状态取最大动作价值
- `done=True` 时，`(1.0 - dones)` 会把自举项归零

## 教学追踪脚本

- [trace_dqn_updates.py](../experiments/05-cartpole-dqn/trace_dqn_updates.py)

运行：

```bash
cd experiments/05-cartpole-dqn
python trace_dqn_updates.py
```

这个脚本不做完整训练，而是固定一个小批量样本，直接打印：

- 每条样本的 `Q_online(s, .)`
- 被选动作的 `Q_pred`
- `target_net` 产生的 `max_next_Q`
- `TD target`
- 一次优化前后的 `loss` 变化

## 对应内容

- [05-cartpole-dqn](../experiments/05-cartpole-dqn/README.md)
- [03-Q-Learning的值传播与Q表更新](./03-Q-Learning的值传播与Q表更新.md)
- [06-n-step-SARSA的多步回报与折中更新](./06-n-step-SARSA的多步回报与折中更新.md)
