---
name: training
description: >
  Training loop, environment simulation, CUDA kernel, training config and known bugs.
  Use when working on training improvements, environment changes, reward tuning, or fixing training code.
argument-hint: "training, env, reward, loss, CUDA kernel, speed_mtp, checkpoint"
---

# 训练系统

## 入口

```bash
uv run bitpilot-train $(cat configs/single_agent.args)
uv run bitpilot-train --help  # 全部参数
```

## 架构

```
train.py  →  Env (env.py)  →  bitpilot._C (CUDA kernel)
   ↓              ↓
  Model        run_forward / run_backward / render / find_nearest_pt
```

- 50000 iterations, batch=64, 150 timesteps/iter
- Adam + CosineAnnealing, lr=0.001
- Gradient: `grad_decay=0.4` (自定义梯度衰减)
- 日志: TensorBoard (`runs/`) + Aim (`share/aim_data/`)

## 训练配置 (single_agent.args)

```
--single --speed_mtp 4 --coef_d_acc 0.01 --coef_d_jerk 0.001
--ground_voxels --random_rotation --yaw_drift
--coef_collide 7.5 --coef_obj_avoidance 3.0
--cam_angle 20 --fov_x_half_tan 0.82
```

## CUDA 扩展 `bitpilot._C`

### run_forward (13 positional args)

```
run_forward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a,
            ctl_dt, airmode)
→ [act_next, p_next, v_next, a_next]
```

- 所有参数 positional-only（不支持 kwargs！）
- ctl_dt: float, airmode: float (不是 int)

### run_backward (14 positional args)

```
run_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
             v, v_wind, act_next,
             d_act_next, d_p_next, d_v_next, d_a_next,
             grad_decay, ctl_dt)
→ [d_act_pred, d_act, d_p, d_v, d_a]
```

### 其他函数

- `render(...)` — 深度图渲染
- `find_nearest_pt(...)` — 碰撞/避障距离
- `update_state_vec(...)` — 偏航状态更新

## 已知 bug (train.py)

1. `torch.cross(up, fwd)` 缺 `dim=-1` (~L188)
2. `barrier()` 维度处理不完整
3. timesteps 硬编码 135 (应从 args.timesteps 读取)
4. 某个 loss 项 `+` 应为 `*`

## 改进计划

见 `plan.md`:
- 距离相关速度衰减 → 学会减速停止
- 到达后保持目标 → 学会悬停 + 悬停避障
