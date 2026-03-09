---
name: gazebo-deploy
description: >
  Gazebo simulation deployment: ROS2 topics, sensor config, coordinate systems, inference node.
  Use when working on sim deployment, flight testing, sensor integration, or debugging flight behavior.
argument-hint: "ROS2 topics, Gazebo, sensor config, inference node, flight"
---

# Gazebo 仿真部署

完整文档在 `README.md` §Gazebo 仿真部署对接指南。以下是快速参考。

## 坐标系

全链路 **FLU** (X前 Y左 Z上)。世界系和机体系都是 FLU，无需坐标变换。

## ROS2 话题

### 输入

| 话题 | 类型 | 频率 | 用途 |
|------|------|------|------|
| `/uav1/rgbd_camera/depth_image` | Image (32FC1) | ~32 Hz | 深度图 |
| `/uav1/odometry` | Odometry | ~53 Hz | 位置/速度/姿态 |
| `/uav1/imu` | Imu | ~250 Hz | 高频姿态(可选) |

### 输出

| 话题 | 类型 | 频率 | QoS |
|------|------|------|-----|
| `/uav1/setpoint_raw/local` | PositionTarget | 15 Hz | BEST_EFFORT |

## 控制接口

- `coordinate_frame = FRAME_BODY_NED` (8) — 体坐标系
- 使用 velocity 字段 (vx, vy, vz) + yaw_rate
- type_mask = 1991 (忽略位置+加速度+绝对航向)

## 相机参数 (已修改)

| 参数 | 值 |
|------|-----|
| hfov | 1.365 rad (~78°) |
| 分辨率 | 640×480 |
| clip_far | 25.0 m |
| 安装 pitch | -20° (上仰) |

## 推理节点

```bash
source /opt/ros/jazzy/setup.bash
uv run bitpilot-infer --target 10 0 1.5
```

入口: `src/bitpilot/deploy/inference_node.py`

关键参数: `--target`, `--checkpoint`, `--max_speed`, `--margin`, `--yaw_rate_gain`

## 航向控制

模型不输出 yaw。推理节点内置 P 控制器 (gain=1.5, max=1.5 rad/s)，根据水平速度方向驱动航向跟随。

## 仿真控制 API

见 `.github/skills/sim-control/SKILL.md`

```bash
curl http://gz-sim:6799/status
curl -X POST http://gz-sim:6799/restart
```
