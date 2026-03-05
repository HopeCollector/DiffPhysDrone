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
├── check_sim.py        # 仿真环境预飞检测脚本
├── env_cuda.py         # 仿真环境 Python 封装
├── main_cuda.py        # 训练主程序入口
├── model.py            # 神经网络模型定义
├── pyproject.toml      # 项目依赖与配置 (uv)
└── README.md           # 项目说明文档
```

---

## 模型推理接口文档（部署参考）

> 本章节详细记录了模型的输入输出规格及数据预处理/后处理流程，供 ROS2 节点等外部系统对接使用。

### 一、模型概述

模型定义在 `model.py` 中，类名 `Model`。它是一个 **视觉-状态融合的循环神经网络**，流程为：

```
深度图 ──→ CNN(stem) ──→ 图像特征(192维)
                                          ├─→ 相加融合 ──→ GRUCell ──→ FC ──→ 动作输出(6维)
状态向量 ──→ Linear(v_proj) ──→ 状态特征(192维)
```

带有 GRU 隐状态，每一步推理需要传入上一步的隐状态（首次传入 `None`）。

### 二、模型实例化

训练使用的实例化方式（见 `main_cuda.py`）：

```python
from model import Model

# 有里程计模式（推荐，部署时通常有 odom）
model = Model(dim_obs=10, dim_action=6)

# 无里程计模式（dim_obs 少 3 维速度）
# model = Model(dim_obs=7, dim_action=6)
```

加载 checkpoint：

```python
state_dict = torch.load('checkpoint0004.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()
```

### 三、模型输入：深度图 `x`

| 属性 | 值 |
|------|----|
| 形状 | `(B, 1, 12, 16)`，B 为 batch size，部署时通常为 1 |
| 数据类型 | `torch.float32` |
| 含义 | 经预处理后的 **视差图** |

#### 预处理流程

原始深度图来自传感器（如 RGBD 相机），需经过以下步骤：

```python
import torch
import torch.nn.functional as F

# 第 1 步：获取原始深度图
# 来自传感器的深度图，单位为米，形状为 (H_raw, W_raw)
# 例如 Gazebo RGBD 相机输出为 (480, 640)，float32

# 第 2 步：Resize 到训练分辨率 48×64
# 训练时环境直接输出 (48, 64) 的深度图
depth = F.interpolate(
    depth_raw[None, None],       # (1, 1, H_raw, W_raw)
    size=(48, 64),
    mode='bilinear',
    align_corners=False
)[0, 0]                          # (48, 64)

# 第 3 步：深度 → 视差变换 + clamp
# 训练代码原文: x = 3 / depth.clamp_(0.3, 24) - 0.6
# 部署时不加噪声
x = 3.0 / depth.clamp(0.3, 24) - 0.6    # (48, 64)

# 第 4 步：4×4 MaxPool
# 训练代码原文: x = F.max_pool2d(x[:, None], 4, 4)
x = F.max_pool2d(x[None, None], 4, 4)    # (1, 1, 12, 16)
```

> **注意**：训练时在视差变换后会加入 `torch.randn_like(depth) * 0.02` 的高斯噪声做数据增强，部署时不需要加噪声。

#### 关于相机 FOV 的匹配

训练时使用的相机 `fov_x_half_tan` 参数控制水平视场角（FOV）：

| 配置 | `fov_x_half_tan` | 对应水平 FOV |
|------|-------------------|-------------|
| 默认 | 0.53 | ~56° |
| single_agent.args | 0.82 | ~78° |
| multi_agent.args | 0.82 | ~78° |

Gazebo 仿真相机 hfov 已修改为 1.365 rad，实测 fx = 393.70，`fov_x_half_tan = 320/393.70 ≈ 0.813`，与训练值 0.82 误差 <1%，**无需裁剪**。

### 四、模型输入：状态向量 `v`

| 属性 | 值 |
|------|----|
| 形状 | `(B, 10)` (有里程计) 或 `(B, 7)` (无里程计) |
| 数据类型 | `torch.float32` |

#### 有里程计模式 `dim_obs=10` 各维度定义

状态向量由以下部分 **按顺序拼接** 而成：

```python
state = torch.cat([
    local_v,          # [0:3]  当前速度（航向坐标系下），3 维
    target_v_local,   # [3:6]  目标速度（航向坐标系下），3 维
    body_up,          # [6:9]  机体 Z 轴在世界系的方向，3 维
    margin,           # [9]    碰撞半径，1 维
], dim=-1)            # 总计 10 维
```

各分量详细说明：

##### (1) `local_v` — 航向坐标系下的当前速度 [0:3]

```python
# env.v 是世界坐标系下的速度 (B, 3)
# R_yaw 是仅保留航向（yaw-only）的旋转矩阵 (B, 3, 3)
local_v = v_world @ R_yaw    # (B, 3)
```

**R_yaw 的构造方法**（关键！训练中剥离了 pitch/roll，只保留 yaw）：

```python
# 从完整姿态矩阵中提取前向（X 轴），投影到水平面
fwd = R_body[:, :, 0].clone()   # 机体 X 轴方向
fwd[:, 2] = 0                   # 去除 Z 分量（投影到水平面）
fwd = F.normalize(fwd, 2, -1)   # 归一化

up = torch.tensor([0., 0., 1.])  # 世界 Z 轴
left = torch.cross(up, fwd)      # 叉乘得到左方向

# 构建 yaw-only 旋转矩阵 [Forward, Left, Up]
R_yaw = torch.stack([fwd, left, up], dim=-1)  # (B, 3, 3)
```

> **R_yaw 的物理意义**：这个旋转矩阵描述了一个"只考虑航向角、忽略俯仰和横滚"的坐标系。它将世界坐标系下的向量变换到以无人机当前航向为 X 轴、世界 Z 轴为上方的水平坐标系中。

##### (2) `target_v_local` — 航向坐标系下的目标速度 [3:6]

```python
# p_target 是目标位置 (B, 3)，p 是当前位置 (B, 3)
target_v_raw = p_target - p

# 限制最大速度
target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
target_v = (target_v_raw / target_v_norm) * torch.minimum(target_v_norm, max_speed)

# 变换到航向坐标系
target_v_local = target_v @ R_yaw    # (B, 3)
```

> **`max_speed` 参数**：训练时随机采样范围为 `0.75 ~ 3.25 * speed_mtp` m/s（其中 `speed_mtp` 可在 args 中配置，single_agent 为 4，默认为 1）。部署时应根据任务需求设定固定值。

##### (3) `body_up` — 机体 Z 轴在世界坐标系的方向 [6:9]

```python
# env.R 是完整的机体姿态矩阵 (B, 3, 3)
# 第 3 列（索引 2）是机体 Z 轴（上方向）在世界坐标系下的表示
body_up = R_body[:, :, 2]    # (B, 3)
```

这个向量反映无人机当前的倾斜姿态。当无人机水平悬停时，`body_up ≈ [0, 0, 1]`。

##### (4) `margin` — 碰撞安全半径 [9]

```python
margin = torch.tensor([0.15])    # 标量，训练时随机范围 [0.1, 0.3]
```

部署时设为常数，根据实际无人机尺寸选择（推荐 0.15m）。

### 五、模型输入：GRU 隐状态 `hx`

| 属性 | 值 |
|------|----|
| 形状 | `(B, 192)` |
| 数据类型 | `torch.float32` |
| 初始值 | `None`（首次推理时传入 `None`，GRUCell 内部会自动初始化为零） |

每次推理后模型会返回更新后的 `hx`，需保存并传入下一步。当任务重启或目标变更时，可重置为 `None`。

### 六、模型输出

```python
act, _, hx = model(x, state, hx)
# act:  (B, 6) — 动作输出
# _:    None（未使用）
# hx:   (B, 192) — 更新后的 GRU 隐状态
```

#### `act` 的后处理：从模型输出到加速度指令

模型输出 `act` 是 **航向坐标系下** 的 6 维向量，需要经过以下步骤转换为世界坐标系下的加速度指令：

```python
# 第 1 步：reshape + 旋转到世界坐标系
# act: (B, 6) → (B, 3, 2)
# R_yaw: (B, 3, 3)  — 同输入预处理中构造的 yaw-only 旋转矩阵
a_pred, v_pred = (R_yaw @ act.reshape(B, 3, 2)).unbind(-1)
# a_pred: (B, 3) — 期望加速度（世界系）
# v_pred: (B, 3) — 速度估计（世界系），辅助输出

# 第 2 步：加速度 → 推力指令（补偿重力）
g = torch.tensor([0., 0., -9.80665])
thr_est_error = 1.0    # 训练时 ~1.0±0.01，部署时固定为 1.0
thrust_cmd = (a_pred - v_pred - g) * thr_est_error + g
# thrust_cmd: (B, 3) — 最终世界坐标系下的加速度指令
```

> **关于 `v_pred`**：这是模型同时学到的一个速度估计头，在推力计算中用于补偿阻力。在部署环境中如果动力学模型与训练差异较大，可以考虑将 `v_pred` 替换为实际测量速度或直接置零观察效果。

### 七、坐标系约定

训练环境使用的坐标系：

```
      Z (上)
      │
      │
      │
      └──── Y (左)
     /
    /
   X (前)
```

- **世界坐标系**: X-前, Y-左, Z-上（FLU, Front-Left-Up）
- **机体坐标系**: 旋转矩阵 `R` 的三列分别为机体的 [Forward, Left, Up] 在世界坐标系中的方向
- **航向坐标系 R_yaw**: 仅保留 yaw 角的旋转矩阵，X 轴为机头水平投影方向，Z 轴始终为世界 Z 轴

部署时需注意 Gazebo/PX4 可能使用 NED（North-East-Down）或 ENU（East-North-Up）坐标系，需做相应转换。

### 八、完整推理伪代码

```python
import torch
import torch.nn.functional as F
from model import Model

# ── 初始化 ────────────────────────────────────────
device = torch.device('cuda')
model = Model(dim_obs=10, dim_action=6).to(device)
model.load_state_dict(torch.load('checkpoint0004.pth', map_location=device))
model.eval()

hx = None                               # GRU 隐状态
g = torch.tensor([0., 0., -9.80665], device=device)
MARGIN = 0.15                            # 碰撞半径（米），根据实际机型调整
MAX_SPEED = 2.0                          # 最大速度（米/秒），根据任务调整

# ── 每个控制周期（约 15 Hz）────────────────────────
@torch.no_grad()
def inference_step(depth_raw, v_world, R_body, p_current, p_target, hx):
    """
    单步推理函数。

    参数:
        depth_raw:  (H, W) float32, 原始深度图，单位米
        v_world:    (3,) float32, 世界坐标系下的线速度
        R_body:     (3, 3) float32, 完整机体姿态旋转矩阵 [fwd, left, up]
        p_current:  (3,) float32, 世界坐标系下的当前位置
        p_target:   (3,) float32, 世界坐标系下的目标位置
        hx:         (1, 192) 或 None, GRU 隐状态

    返回:
        thrust_cmd: (3,) float32, 世界坐标系下的加速度指令
        hx:         (1, 192), 更新后的 GRU 隐状态
    """
    # ── 深度图预处理 ──
    depth = F.interpolate(depth_raw[None, None], size=(48, 64),
                          mode='bilinear', align_corners=False)[0, 0]
    x = 3.0 / depth.clamp(0.3, 24) - 0.6
    x = F.max_pool2d(x[None, None], 4, 4)                # (1, 1, 12, 16)

    # ── 构建 R_yaw ──
    fwd = R_body[:, 0].clone()
    fwd[2] = 0.0
    fwd = F.normalize(fwd, dim=0)
    up = torch.tensor([0., 0., 1.], device=fwd.device)
    left = torch.cross(up, fwd)
    R_yaw = torch.stack([fwd, left, up], dim=-1)          # (3, 3)

    # ── 构建状态向量 ──
    local_v = v_world @ R_yaw                              # (3,)
    target_v_raw = p_target - p_current
    target_v_norm = target_v_raw.norm()
    target_v = (target_v_raw / target_v_norm) * min(target_v_norm.item(), MAX_SPEED)
    target_v_local = target_v @ R_yaw                      # (3,)
    body_up = R_body[:, 2]                                 # (3,)
    margin = torch.tensor([MARGIN], device=device)

    state = torch.cat([local_v, target_v_local, body_up, margin]).unsqueeze(0)  # (1, 10)

    # ── 模型推理 ──
    act, _, hx = model(x, state, hx)                       # act: (1, 6)

    # ── 后处理 ──
    R_yaw_b = R_yaw.unsqueeze(0)                           # (1, 3, 3)
    a_pred, v_pred = (R_yaw_b @ act.reshape(1, 3, 2)).unbind(-1)
    thrust_cmd = (a_pred - v_pred - g) + g                 # thr_est_error=1.0
    thrust_cmd = thrust_cmd.squeeze(0)                     # (3,)

    return thrust_cmd, hx
```

### 九、关键超参数速查表

| 参数 | 训练值 | 部署建议 | 说明 |
|------|--------|----------|------|
| 控制频率 | 15 Hz (±10%) | 15 Hz 定时器 | 模型设计的控制周期 |
| 深度图原始分辨率 | 48×64 (直出) | 需从传感器 resize 到 48×64 | 训练环境直接渲染此分辨率 |
| `fov_x_half_tan` | 0.53 或 0.82 | 需匹配或裁剪 | 相机水平半视角正切值 |
| `cam_angle` | 10° 或 20° | 物理安装角度匹配 | 相机俯仰角（向上为正，补偿前倾） |
| `margin` | 随机 [0.1, 0.3] | 固定 0.15 | 碰撞安全半径(米) |
| `max_speed` | 随机 [0.75, 3.25]×speed_mtp | 固定值，如 2.0 | 目标速度上限(m/s) |
| `thr_est_error` | 随机 ~1.0±0.01 | 1.0 | 推力估计误差补偿 |
| `dim_obs` | 10 (odom) / 7 (no_odom) | 10 | 状态向量维度 |
| `dim_action` | 6 | 6 | 模型输出维度 |
| GRU hidden size | 192 | 192 | 不可修改，与权重绑定 |

---

## Gazebo 仿真部署对接指南

> 本章节记录 Gazebo 仿真环境的已确认配置，以及模型输出到实际飞控指令的完整对接方案。

### 一、仿真环境已确认配置

#### 1.1 RGBD 相机参数（已修改后的实际值）

| 参数 | 值 | 说明 |
|------|----|------|
| `rgbd_camera_rate` | 30 Hz | 深度图发布频率（实测 ~32 Hz） |
| `rgbd_camera_hfov` | **1.365 rad (~78°)** | 水平视场角，已匹配训练 `fov_x_half_tan=0.82` |
| `rgbd_camera_width` | 640 px | 图像宽度 |
| `rgbd_camera_height` | 480 px | 图像高度 |
| `rgbd_camera_clip_near` | 0.1 m | 最近裁剪距离 |
| `rgbd_camera_clip_far` | **25.0 m** | 最远裁剪距离，已覆盖训练 clamp 上限 24m |
| 深度编码 | `32FC1` | float32，单位米，超出 clip_far 为 inf |
| 相机安装 pitch | **-20°（上仰 20°）** | 已匹配训练 `cam_angle=20`，补偿前倾下压 |
| 实测内参 fx | 393.70 | `fov_x_half_tan = 320/393.70 ≈ 0.813`，与训练 0.82 误差 <1% |

#### 1.2 坐标系约定

| 坐标系 | 约定 | 说明 |
|--------|------|------|
| 世界坐标系 | **FLU** (X前 Y左 Z上) | 原点在无人机默认起飞位置 |
| 机体坐标系 | **FLU** (X前 Y左 Z上) | 与训练环境一致，**无需坐标变换** |

#### 1.3 控制接口

| 项目 | 值 |
|------|----|
| 话题 | `/uav1/setpoint_raw/local` |
| 消息类型 | `mavros_msgs/msg/PositionTarget` |
| `coordinate_frame` | `FRAME_BODY_NED` (8) — 机体坐标系 |
| 控制模式 | **本体坐标系下的速度控制** |
| QoS | BEST_EFFORT |

#### 1.4 ROS2 话题总览

**订阅（输入）：**

| 话题 | 类型 | 频率 | QoS | 用途 |
|------|------|------|-----|------|
| `/uav1/rgbd_camera/depth_image` | `sensor_msgs/msg/Image` | ~32 Hz | RELIABLE | 深度图 → 模型视觉输入 |
| `/uav1/odometry` | `nav_msgs/msg/Odometry` | ~53 Hz | RELIABLE | 位置 `p`、速度 `v`、姿态 `R` |
| `/uav1/imu` | `sensor_msgs/msg/Imu` | ~250 Hz | RELIABLE | 姿态四元数（高频、可选） |
| `/uav1/collision` | `uav_interfaces/msg/Collision` | 按需 | BEST_EFFORT | 碰撞检测（可用于评估） |

**发布（输出）：**

| 话题 | 类型 | 频率 | QoS | 用途 |
|------|------|------|-----|------|
| `/uav1/setpoint_raw/local` | `mavros_msgs/msg/PositionTarget` | 15 Hz | BEST_EFFORT | 机体坐标系速度指令 |

### 二、关键对接问题与解决方案

#### 2.1 模型输出是加速度，飞控接口是速度 — 如何对接？

模型输出 `thrust_cmd` 是世界坐标系下的 **加速度指令** (m/s²)，而飞控接受本体坐标系下的 **速度指令** (m/s)。需要做两步转换：

```python
# 第 1 步：加速度积分为速度（世界坐标系）
# v_cmd_world = v_current_world + thrust_cmd * dt
# 其中 dt = 1/15 (控制周期)
dt = 1.0 / 15.0
v_cmd_world = v_current_world + thrust_cmd * dt    # (3,)

# 第 2 步：世界坐标系速度 → 机体坐标系速度
# R_body: (3, 3) 完整机体姿态旋转矩阵（从 odometry 的 quaternion 得到）
# R_body 的列是机体轴在世界系的方向 → R_body^T 将世界系向量变换到机体系
v_cmd_body = R_body.T @ v_cmd_world                # (3,)
```

然后填充 `PositionTarget`：

```python
from mavros_msgs.msg import PositionTarget

msg = PositionTarget()
msg.header.stamp = node.get_clock().now().to_msg()
msg.coordinate_frame = PositionTarget.FRAME_BODY_NED  # 8, 体坐标系
msg.type_mask = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |  # 忽略位置
    PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |  # 忽略加速度
    PositionTarget.IGNORE_YAW  # 忽略绝对航向，使用 yaw_rate
)
# type_mask = 0b0000_0111_11_000_111 = 0x07C7 = 1991
msg.velocity.x = float(v_cmd_body[0])   # 前进速度 (FLU: X 前)
msg.velocity.y = float(v_cmd_body[1])   # 横向速度 (FLU: Y 左)
msg.velocity.z = float(v_cmd_body[2])   # 垂直速度 (FLU: Z 上)
msg.yaw_rate = 0.0                       # 航向角速率，见下方说明
```

> **关于航向控制**：模型本身不直接输出航向指令。训练中航向由 `v_pred`（速度估计）驱动，机头跟随飞行方向转向。部署时可设 `yaw_rate = 0`（保持当前航向），或根据 `v_pred` 的水平分量方向计算期望 yaw_rate 来实现随动转向。

#### 2.2 深度图解析与 inf 处理

深度图编码为 `32FC1`（float32，单位米）。超出 `clip_far` 或无反射面的像素值为 `inf`，需在预处理时替换：

```python
import numpy as np

# 从 ROS Image 消息解析出 float32 深度图
depth_np = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

# 将 inf/nan 替换为 clip_far 值（当前 25m，需 ≥ 训练 clamp 上限 24m）
depth_np = np.nan_to_num(depth_np, nan=25.0, posinf=25.0, neginf=0.1)

depth = torch.from_numpy(depth_np).to(device)
```

> 相机 FOV、安装俯仰角、clip_far 等传感器参数的要求值见 [1.1 RGBD 相机参数](#11-rgbd-相机参数已修改后的实际值)。若 FOV 不匹配且无法修改硬件，可在软件中对深度图做中心裁剪：
> ```python
> crop_ratio = training_fov_half_tan / sim_fov_half_tan
> crop_w = int(width * crop_ratio)
> crop_h = int(height * crop_ratio)
> depth_cropped = depth_raw[
>     (height - crop_h) // 2 : (height + crop_h) // 2,
>     (width - crop_w) // 2 : (width + crop_w) // 2
> ]
> ```

#### 2.3 里程计数据提取

`/uav1/odometry` (`nav_msgs/msg/Odometry`) 提供以下信息：

```python
from scipy.spatial.transform import Rotation

# ── 位置（世界坐标系 FLU）──
p = torch.tensor([
    msg.pose.pose.position.x,
    msg.pose.pose.position.y,
    msg.pose.pose.position.z,
], device=device)

# ── 姿态四元数 → 旋转矩阵 ──
q = msg.pose.pose.orientation
R_body = torch.from_numpy(
    Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
).float().to(device)    # (3, 3), 列为 [fwd, left, up] 在世界系的方向

# ── 速度 ──
# nav_msgs/Odometry 的 twist 在 child_frame (body frame) 中
# 需要转到世界坐标系供模型使用
v_body = torch.tensor([
    msg.twist.twist.linear.x,
    msg.twist.twist.linear.y,
    msg.twist.twist.linear.z,
], device=device)
v_world = R_body @ v_body    # (3,)

# ── body_up 向量（模型状态输入需要）──
body_up = R_body[:, 2]       # 旋转矩阵第 3 列
```

### 三、推荐的 ROS2 节点架构

```
┌──────────────────────────────────────────────────────────┐
│                    inference_node (15 Hz)                 │
│                                                          │
│   Subscribers:                                           │
│     /uav1/rgbd_camera/depth_image  ──→ 缓存最新深度图    │
│     /uav1/odometry                 ──→ 缓存最新状态       │
│                                                          │
│   Timer (15 Hz):                                         │
│     1. 读取缓存的深度图 + odometry                        │
│     2. 深度图预处理 (crop/resize → 视差 → maxpool)        │
│     3. 构造状态向量 (local_v, target_v, body_up, margin)  │
│     4. model.forward(x, state, hx) → act                │
│     5. 后处理 (加速度 → 积分 → 体坐标系速度)              │
│     6. 发布 PositionTarget 到 /uav1/setpoint_raw/local   │
│                                                          │
│   Publisher:                                             │
│     /uav1/setpoint_raw/local       ←── PositionTarget    │
│                                                          │
│   Parameters:                                            │
│     target_position: [x, y, z]     ←── 目标位置           │
│     max_speed: 2.0                                       │
│     margin: 0.15                                         │
│     checkpoint: checkpoint0004.pth                       │
└──────────────────────────────────────────────────────────┘
```

**关键设计要点：**
- 深度图和 odom 回调只做缓存，不做推理（避免频率耦合）
- 推理由独立的 15 Hz 定时器驱动（匹配训练控制频率）
- GRU 隐状态 `hx` 在节点内部持久保存，目标点变更时可选择重置
- 所有 tensor 运算在 GPU 上完成，仅在发布消息时转回 CPU

### 四、仿真环境配置修改清单

以下参数已在 Gazebo Jinja 模板中修改完成（✅ 已通过 `check_sim.py` 验证）：

| # | 参数 | 原始值 | 已改为 | 验证结果 |
|---|------|--------|--------|----------|
| 1 | `rgbd_camera_hfov` | 1.57 rad (90°) | **1.365 rad** (~78°) | ✅ 实测 `fov_x_half_tan=0.813`，误差 <1% |
| 2 | 相机安装 pitch | 0° (正前方) | **-20°** (上仰 20°) | ✅ 已在模板中确认 |
| 3 | `rgbd_camera_clip_far` | 10.0 m | **25.0 m** | ✅ 实测最大有效深度 24.8m |

其余参数（分辨率 640×480、帧率 30Hz、clip_near 0.1m）无需修改。

### 五、仿真环境预飞检测

修改完成后使用 `check_sim.py` 验证所有配置：

```bash
source /opt/ros/jazzy/setup.bash
python3 check_sim.py [--timeout 5]
```

检测内容（28 项）：

| 类别 | 检测项 |
|------|--------|
| 话题 | depth_image / camera_info / odometry / imu 是否在发布 |
| 频率 | 深度图 ≥15Hz、里程计 ≥30Hz、IMU ≥100Hz |
| 深度图 | 分辨率、编码(32FC1)、数据完整性、有效深度范围 |
| 相机 | 内参 fx/fy、FOV 匹配（与 `fov_x_half_tan=0.82` 对比）、宽高比 |
| 里程计 | frame_id、四元数有效性、坐标系 FLU 验证（body_up 方向）、位置、速度 |
| IMU | 四元数有效性、重力方向验证（确认 FLU 坐标系） |
| 控制 | setpoint 话题 QoS 提醒 |

输出示例：
```
========================================================================
  仿真环境预飞检测报告
========================================================================
  总计 28 项:  26 通过  |  2 警告  |  0 失败
  △ 检测基本通过, 有 2 项警告建议关注
------------------------------------------------------------------------
```

> 2 个 WARN 为预期：相机安装角度无法从话题自动验证（需人工确认模板）、setpoint 话题 QoS 提醒。
