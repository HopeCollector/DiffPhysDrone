# Vision-based Agile Flight Training Code

## 概览

本仓库包含 **Learning Vision-based Agile Flight via Differentiable Physics** 的训练代码。

**更新说明**：
本项目已针对 **NVIDIA RTX 5090** 显卡进行了适配与测试。同时，项目依赖管理已迁移至 `uv`，并调整了部分文件结构。

## 环境配置

### Python 环境

本项目使用 `uv` 进行高效的包管理和环境配置。请确保您的系统中已安装 `uv`。

1. 安装 `uv` (如果尚未安装):
   ```bash
   pip install uv
   ```

2. 同步项目依赖并创建虚拟环境:
   ```bash
   uv sync
   ```

3. 激活虚拟环境:
   ```bash
   source .venv/bin/activate
   ```

### 编译 CUDA 算子

由于项目结构调整，CUDA 算子的安装目标已变更为 `quadsim_cuda` 目录。请在激活的虚拟环境中运行以下命令进行编译和安装：

```bash
uv pip install -e quadsim_cuda
```

## 训练

训练流程与原版保持一致。使用以下命令启动训练：

```bash
# 多智能体训练 (Multi-agent)
python main_cuda.py $(cat configs/multi_agent.args)

# 单智能体训练 (Single-agent)
python main_cuda.py $(cat configs/single_agent.args)
```

## 评估

若要在多智能体设置中评估训练好的模型，请按以下步骤操作：

1. 启动模拟器 (需使用配套的模拟器程序):
   ```bash
   cd <path to multi agent code supplementary>
   ./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
   ```

2. 运行评估脚本:
   ```bash
   python eval.py --resume <path to checkpoint> --target_speed 2.5
   ```

## 项目结构

```plaintext
.
├── configs/            # 训练配置文件 (.args)
├── quadsim_cuda/       # 四旋翼动力学仿真 CUDA 算子包 (需编译安装)
├── runs/               # 训练运行日志 (TensorBoard) 和模型检查点
├── src/                # 原始源码目录 (部分功能已迁移至 quadsim_cuda)
├── env_cuda.py         # 仿真环境 Python 封装
├── main_cuda.py        # 训练主程序入口
├── model.py            # 神经网络模型定义
├── pyproject.toml      # 项目依赖与配置 (uv)
└── README.md           # 项目说明文档
```
