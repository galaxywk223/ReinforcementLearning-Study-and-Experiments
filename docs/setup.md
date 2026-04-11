# Setup

这份说明只覆盖当前仓库已经公开的内容，即表格型强化学习项目和对应教程文档。

## 1. 推荐方式：使用 Conda

```bash
conda env create -f environment.yml
conda activate ReinforcementLearning
```

当前 `environment.yml` 只包含运行现有公开项目所需的最小依赖：

- `python`
- `numpy`
- `matplotlib`
- `gymnasium`
- `tqdm`

## 2. 备用方式：直接安装项目依赖

如果你不使用 `conda`，也可以在自己的 Python 环境里安装：

```bash
pip install -r projects/requirements.txt
```

## 3. 运行当前项目

```bash
cd projects/frozenlake-tabular-q
python train.py --episodes 4000 --render-final-policy
```

或者运行逐步追踪脚本：

```bash
cd projects/frozenlake-tabular-q
python trace_q_updates.py --episodes 6
```

## 4. Windows / CPU / GPU 说明

- 当前公开项目是表格型方法，CPU 就可以运行
- 当前仓库不要求安装 PyTorch，也不要求 CUDA
- 如果后续新增 `DQN`、`PPO` 等深度强化学习实验，再单独补充 GPU 依赖说明更合适

## 5. 常见问题

### `gymnasium` 导入失败

优先确认你是否已经激活正确环境：

```bash
conda activate ReinforcementLearning
```

### 图像不显示或无法保存

确认 `matplotlib` 已正确安装，并检查当前目录是否有写入权限。
