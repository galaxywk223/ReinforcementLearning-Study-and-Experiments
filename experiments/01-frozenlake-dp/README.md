# FrozenLake 动态规划

这个目录保存 `FrozenLake-v1` 上的动态规划代码和最小运行说明，用于展示已知模型条件下的策略评估、策略迭代与价值迭代。

## 关联笔记

- [03-动态规划的策略评估、策略迭代与价值迭代](../../notes/03-动态规划的策略评估、策略迭代与价值迭代.md)

## 实验内容

- 同时运行策略迭代与价值迭代的完整入口 `train.py`
- 打印前几轮价值迭代 Bellman sweep 的教学脚本 `trace_dp_iterations.py`
- 收敛曲线、策略网格与训练摘要导出

## 代表结果

- 评估平均奖励：`0.755`
- 评估成功率：`0.755`
- 策略迭代改进轮数：`7`
- 价值迭代 Bellman sweep 数：`438`

<p align="center">
  <img src="../../assets/figures/frozenlake-dp/convergence_curve.png" alt="FrozenLake 动态规划收敛曲线" width="920" />
</p>

## 运行命令

```bash
cd experiments/01-frozenlake-dp
python train.py --render-final-policy
python train.py --non-slippery --run-name deterministic
python trace_dp_iterations.py --iterations 5
```

## 输出目录

- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/convergence_curve.png`
- `outputs/<run_name>/policy_grid.png`

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train.py` | 策略迭代与价值迭代的完整入口 |
| `trace_dp_iterations.py` | 打印前几轮价值迭代状态值变化 |
