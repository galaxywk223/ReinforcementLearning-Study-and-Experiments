# Documentation

这个目录收录仓库的公开教程文档，按“入门问题 -> 基础概念 -> 表格方法”组织。

## Getting Started

- [第一次理解强化学习](getting-started/first-questions.md)

## Foundations

- [MDP、回报与 Bellman 方程](foundations/mdp-and-bellman.md)

## Tabular Methods

- [Q-Learning 是怎么一步步把 Q 表学出来的](tabular/q-learning-step-by-step.md)
- [SARSA 是怎么用“下一步真实动作”更新 Q 表的](tabular/sarsa-step-by-step.md)
- [SARSA 和 Q-Learning 在 CliffWalking 里会学出什么区别](tabular/sarsa-vs-q-learning.md)

## 建议阅读顺序

1. 先读 `getting-started/first-questions.md`
2. 再读 `foundations/mdp-and-bellman.md`
3. 配合 `projects/frozenlake-tabular-q/` 阅读 `tabular/q-learning-step-by-step.md`
4. 再配合 `projects/cliffwalking-tabular-sarsa/` 阅读 `tabular/sarsa-step-by-step.md`
5. 最后阅读 `tabular/sarsa-vs-q-learning.md`，对比两种表格控制方法
