---
name: model-interface
description: >
  Model input/output specification, preprocessing, postprocessing.
  Use when working on inference, deployment, model integration, or debugging model I/O.
argument-hint: "model inputs, outputs, preprocessing, state vector, depth map"
---

# 模型推理接口规格

完整文档在 `README.md` §模型推理接口文档。以下是快速参考。

## 模型签名

```python
from bitpilot.model import Model
model = Model(dim_obs=10, dim_action=6)  # hidden=192
act, _, hx = model(x, state, hx)
# x:     (B, 1, 12, 16) float32 — 视差图
# state: (B, 10) float32         — 状态向量
# hx:    (B, 192) or None        — GRU 隐状态
# act:   (B, 6) float32          — 动作输出
```

## 深度图预处理

```
raw depth (H,W) float32 m
  → resize to (48, 64) bilinear
  → disparity: 3.0 / clamp(0.3, 24) - 0.6
  → max_pool2d(4,4) → (1, 1, 12, 16)
```

## 状态向量 (dim=10)

| 索引 | 名称 | 描述 |
|------|------|------|
| 0:3 | local_v | 航向系当前速度 = v_world @ R_yaw |
| 3:6 | target_v_local | 航向系目标速度 = clamp(target - pos, max_speed) @ R_yaw |
| 6:9 | body_up | R_body[:, 2] 机体Z轴世界方向 |
| 9 | margin | 碰撞安全半径 (推荐 0.15m) |

## R_yaw 构造

```python
fwd = R_body[:, :, 0].clone(); fwd[:, 2] = 0; fwd = normalize(fwd)
up = [0,0,1]; left = cross(up, fwd)
R_yaw = stack([fwd, left, up], dim=-1)
```

## 动作后处理

```python
a_pred, v_pred = (R_yaw @ act.reshape(B, 3, 2)).unbind(-1)
g = [0, 0, -9.80665]
thrust_cmd = (a_pred - v_pred - g) * 1.0 + g  # 世界系加速度
v_cmd = v_current + thrust_cmd * dt            # 积分得速度
v_body = R_body.T @ v_cmd                      # 转机体系
```

## 关键超参

| 参数 | 值 |
|------|-----|
| 控制频率 | 15 Hz |
| GRU hidden | 192 |
| fov_x_half_tan | 0.82 (single_agent) |
| cam_angle | 20° |
| max_speed 部署建议 | 2.0 m/s |
