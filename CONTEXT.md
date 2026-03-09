# BitPilot — 项目上下文

> **读者**：AI 编程助手（多会话协作）。本文件是新会话的首要入口，提供项目全貌和已知陷阱，帮助你快速定位到需要深入阅读的文档。
>
> **维护**：每次重大变更后由修改者负责更新此文件。

---

## 1. 项目是什么

BitPilot 是一个**基于可微物理的视觉敏捷飞行框架**，服务于 skybit 无人机集群项目的单机安全控制核心。

核心流程：`可微仿真 (bitpilot._C) → 端到端训练 (train.py) → ROS2 部署 (inference_node.py) → Gazebo/实机`

部署目标硬件：NVIDIA Orin NX 16G。

## 2. 代码地图

```
src/bitpilot/
├── _csrc/                 # C++/CUDA 源码 → 编译为 bitpilot._C (仅渲染/碰撞)
│   ├── quadsim.cpp        #   pybind11 绑定入口
│   ├── quadsim_kernel.cu  #   渲染/碰撞检测 kernel
├── model.py               # CNN+GRU 网络 (59 行，dim_obs=10, dim_action=6, hidden=192)
├── env.py                 # 可微仿真环境 Env 类 (~420 行)，动力学为纯 PyTorch
├── train.py               # 训练主循环 (368 行)，入口 bitpilot-train
└── deploy/
    └── inference_node.py  # ROS2 推理节点 (517 行)，入口 bitpilot-infer

tests/
├── test_model.py          # 模型单元测试 (7 tests) ✅
├── test_env.py            # 仿真环境测试 (10 tests) ✅
├── test_cuda_gradient.py  # 梯度验证 (2 tests) ✅
├── test_cuda_dynamics_forward.py  # 动力学前向测试 (12 tests) ✅
├── test_cuda_dynamics_backward.py # 动力学反向测试 (3 tests) ✅
├── test_cuda_collision.py     # 碰撞/状态更新测试 (14 tests) ✅
├── test_cuda_render.py        # 渲染 kernel 测试 (14 tests) ✅
├── test_preflight.py      # Gazebo 预飞检测 (需仿真运行)
└── test_flight.py         # 飞行集成测试 (需仿真运行)

configs/
├── single_agent.args      # 单机训练参数 (speed_mtp=4, cam_angle=20, fov=0.82)
└── multi_agent.args       # 多机训练参数 (gate 模式)

checkpoints/               # 训练快照 checkpoint0000-0004.pth
```

### 关键文档索引

| 需要了解 | 去看 |
|----------|------|
| 环境搭建、训练/推理命令 | `README.md` §快速开始 |
| 模型输入输出规格、预处理细节 | `README.md` §模型推理接口文档 |
| Gazebo 仿真对接、ROS2 话题 | `README.md` §Gazebo 仿真部署对接指南 |
| 训练改进计划（悬停+避障） | `plan.md` |
| Gazebo 仿真 HTTP 控制 API | `.github/skills/sim-control/SKILL.md` |
| 系统环境、CUDA、Python 版本 | `.github/copilot-instructions.md` |

## 3. 技术架构要点

### 3.1 模型

- **结构**：深度图 → 4层CNN(1→32→64→128→192) → +状态Linear(10→192) → GRUCell(192) → FC(192→6)
- **输入**：视差图 `(B,1,12,16)` + 状态向量 `(B,10)` + GRU隐状态 `(B,192)`
- **输出**：6维动作（航向系下 [a_pred(3), v_pred(3)]）→ 后处理得世界系加速度
- **控制频率**：15 Hz

### 3.2 坐标系

全链路 **FLU**（Front-Left-Up, X前Y左Z上）。Gazebo 仿真也是 FLU，无需坐标变换。

### 3.3 CUDA 扩展 `bitpilot._C`

 kernel (`dynamics_kernel.cu`) 已迁移为纯 PyTorch（`env.py` 中的 `run_forward()` 和 `Env.update_state_vec()`）。

CUDA 扩展现仅保留 3 个渲染/碰撞函数：`render`, `find_nearest_pt`, `rerender_backward`

- 所有参数 **positional-only**（C++ pybind11 绑定，不支持 kwargs）

### 3.4 构建系统

```
pyproject.toml          # 项目元数据、依赖、CLI 入口点
setup.py                # 仅 CUDA 扩展编译 (CUDAExtension → bitpilot._C)
```

- 包管理：`uv`（不是 pip）
- 一键安装：`uv sync`
- `no-build-isolation-package = ["bitpilot"]` — 编译时直接用 venv 的 torch（见 §4 陷阱 #1）

## 4. 已知陷阱（踩过的坑）

### #1 CUDA 扩展 ABI 不兼容

**症状**：`ImportError: undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`

**原因**：PEP 517 构建隔离会在沙箱里下载一个独立的 torch 来编译 C++ 扩展。这个 torch 与 venv 里实际运行的 `torch+cu128` ABI 不同。

**解决**：`pyproject.toml` 中配置 `no-build-isolation-package = ["bitpilot"]`，让 `uv sync` 直接用 venv 的 torch 编译。

**规则**：**永远不要** `uv pip install -e .` 或 `pip install -e .`（会触发隔离构建）。始终用 `uv sync`。

### #2 `bitpilot._C` 函数是 positional-only

**症状**：`TypeError` 或参数传错位

**原因**：pybind11 绑定默认不导出参数名。剩余的 `render`/`find_nearest_pt`/`rerender_backward` 仍为 positional-only。

**规则**：调用时必须按位置传参，不能用 `**kwargs`。

### #3 模型不会悬停

**症状**：无人机到达目标附近后在 1m↔3.7m 间震荡打转

**原因**：训练时 `speed_mtp=4`，目标速度范围 [3, 13] m/s，模型从未见过"停下来"的场景。

**计划**：见 `plan.md` — 引入距离相关速度衰减 + 到达后保持目标继续训练。

### #4 航向控制需要外挂 P 控制器

**症状**：模型不输出 yaw 指令，无人机航向不变，侧飞

**解决**：`inference_node.py` 中实现了 yaw-following P 控制器（gain=1.5, max=1.5 rad/s），根据水平速度方向驱动航向。

### #5 ROS2 在 venv 中需要额外依赖

**症状**：`source /opt/ros/jazzy/setup.bash` 后 `import rclpy` 报 lark/yaml 缺失

**解决**：`pyproject.toml` 已包含 `pyyaml` 和 `lark` 依赖。

### #6 训练代码中的已知 bug（尚未修复）

以下 bug 在 `src/bitpilot/train.py` 中，不影响现有 checkpoint 的使用，但后续训练改进前需修复：

1. **R_yaw 构造 bug** (~L188)：`torch.cross(up, fwd)` 缺少 `dim=-1` 参数（torch 已 deprecate 无 dim 版本）
2. **barrier 维度 bug**：`barrier()` 函数缺少对多维 tensor 的正确处理
3. **魔法数 135**：timesteps 相关硬编码，应改为从 args 读取
4. **`+` vs `*` 笔误**：某个 loss 项用了加法而非乘法

## 5. 开发工作流

### 常用命令

```bash
uv sync                                          # 安装依赖 + 编译 CUDA 扩展
uv run pytest tests/ -v --ignore=tests/test_flight.py --ignore=tests/test_preflight.py  # 单元测试 (92 tests)
uv run bitpilot-train $(cat configs/single_agent.args)     # 训练
uv run bitpilot-train --help                               # 查看所有训练参数

source /opt/ros/jazzy/setup.bash                  # 以下命令需要 ROS2
uv run bitpilot-infer --target 10 0 1.5           # 推理部署
curl http://gz-sim:6799/status                    # 查看仿真状态
curl -X POST http://gz-sim:6799/restart           # 重启仿真
```

### 重编译 CUDA 扩展

如果修改了 `src/bitpilot/_csrc/` 下的 C++/CUDA 源码：

```bash
rm -f src/bitpilot/_C*.so     # 删除旧 .so
uv sync                       # 重新编译
```

### 测试分类

| 标记 | 含义 | 需要 |
|------|------|------|
| (无) | 纯单元测试 | GPU |
| `@pytest.mark.gpu` | CUDA kernel 测试 | GPU |
| `@pytest.mark.preflight` | 仿真预飞检测 | Gazebo 运行 |
| `@pytest.mark.flight` | 飞行集成测试 | Gazebo 运行 |

## 6. 当前开发方向

| 方向 | 状态 | 相关文件 |
|------|------|----------|
| 训练改进（悬停+避障） | 📋 已规划 | `plan.md` |
| 训练代码 bug 修复 | 📋 已识别 | `src/bitpilot/train.py` (见 §4.6) |
| 多机集群协调 | 🔮 未来 | `configs/multi_agent.args` |
| Orin NX 实机部署 | 🔮 未来 | — |
