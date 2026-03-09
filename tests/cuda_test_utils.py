"""
cuda_test_utils — CUDA 测试共享工具函数和 fixtures
==================================================

提供:
  - 旋转矩阵构造工具
  - 简单场景构造工具
  - 正交性/行列式验证
  - 含 airmode 的 PyTorch 参考实现
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

GRAVITY = 9.80665
G_VEC = torch.tensor([0.0, 0.0, -GRAVITY])
CTL_DT = 1.0 / 15.0
GRAD_DECAY = 0.4


# ═══════════════════════════════════════════════════════════
#  旋转矩阵工具
# ═══════════════════════════════════════════════════════════

def make_identity_R(B, device="cuda", dtype=torch.float32):
    """生成 B 个单位旋转矩阵 (B, 3, 3)"""
    return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()


def make_rotation_x(angle_rad, B=1, device="cuda", dtype=torch.float32):
    """绕 X 轴旋转 angle_rad 弧度"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], device=device, dtype=dtype)
    return R.unsqueeze(0).expand(B, -1, -1).contiguous()


def make_rotation_z(angle_rad, B=1, device="cuda", dtype=torch.float32):
    """绕 Z 轴旋转 angle_rad 弧度"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], device=device, dtype=dtype)
    return R.unsqueeze(0).expand(B, -1, -1).contiguous()


def make_random_rotation(B, device="cuda", dtype=torch.float32):
    """通过 QR 分解生成随机正交旋转矩阵 (B, 3, 3)，det=+1"""
    A = torch.randn(B, 3, 3, device=device, dtype=dtype)
    Q, R_ = torch.linalg.qr(A)
    # 确保 det=+1
    det = torch.det(Q)
    Q[det < 0] *= -1
    return Q.contiguous()


# ═══════════════════════════════════════════════════════════
#  验证工具
# ═══════════════════════════════════════════════════════════

def assert_orthogonal(R, atol=1e-5):
    """验证旋转矩阵 R (B,3,3) 满足 R^T R ≈ I 且 det ≈ 1"""
    B = R.shape[0]
    I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, -1, -1)
    RtR = R.transpose(-1, -2) @ R
    assert torch.allclose(RtR, I, atol=atol), (
        f"R^T @ R 不是单位矩阵, max error = {(RtR - I).abs().max().item():.2e}"
    )
    det = torch.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=atol), (
        f"det(R) != 1, values: {det}"
    )


def assert_finite(tensor, name="tensor"):
    """验证张量中无 NaN 或 Inf"""
    assert torch.isfinite(tensor).all(), f"{name} 包含 NaN 或 Inf"


# ═══════════════════════════════════════════════════════════
#  简单场景构造
# ═══════════════════════════════════════════════════════════

def make_empty_scene(B, device="cuda", dtype=torch.float32):
    """构造一个远离原点的空场景（障碍物在 x=-100 处），便于测试"""
    balls = torch.zeros(B, 1, 4, device=device, dtype=dtype)
    balls[..., 0] = -100  # 球心远离
    balls[..., 3] = 0.1   # 小半径

    cylinders = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    cylinders[..., 0] = -100

    cylinders_h = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    cylinders_h[..., 0] = -100

    voxels = torch.zeros(B, 1, 6, device=device, dtype=dtype)
    voxels[..., 0] = -100
    voxels[..., 3:] = 0.1

    return balls, cylinders, cylinders_h, voxels


def make_single_ball_scene(B, ball_center, ball_radius, device="cuda", dtype=torch.float32):
    """构造只有一个球的场景，其余障碍物远离"""
    balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
    balls[:, 0, :3] = torch.tensor(ball_center, device=device, dtype=dtype)
    balls[:, 0, 3] = ball_radius
    return balls, cylinders, cylinders_h, voxels


def make_single_cylinder_scene(B, cyl_x, cyl_y, cyl_r, device="cuda", dtype=torch.float32):
    """构造只有一个垂直圆柱的场景"""
    balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
    cylinders[:, 0, 0] = cyl_x
    cylinders[:, 0, 1] = cyl_y
    cylinders[:, 0, 2] = cyl_r
    return balls, cylinders, cylinders_h, voxels


def make_single_voxel_scene(B, center, half_extents, device="cuda", dtype=torch.float32):
    """构造只有一个 AABB 直方体的场景"""
    balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
    voxels[:, 0, :3] = torch.tensor(center, device=device, dtype=dtype)
    voxels[:, 0, 3:] = torch.tensor(half_extents, device=device, dtype=dtype)
    return balls, cylinders, cylinders_h, voxels


# ═══════════════════════════════════════════════════════════
#  PyTorch 参考实现（含 airmode）
# ═══════════════════════════════════════════════════════════

class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


g_decay = GDecay.apply


def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
                        act_pred, act, p, v, v_wind, a, ctl_dt):
    """
    纯 PyTorch 实现的 run_forward，作为参考基线。

    注意: CUDA 内核中 ctl_dt 是 float (32-bit)，
    因此在 Python 中需要先截断为 float32 精度以匹配。
    """
    # 匹配 CUDA 内核的 float32 精度
    ctl_dt = float(torch.tensor(ctl_dt, dtype=torch.float32).item())
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha

    # 体轴速度分量
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)

    # 阻力：二次项 + 一次项
    drag = drag_2[:, :1] * (
        v_fwd_s.abs() * v_fwd_s * R[..., 0]
        + v_left_s.abs() * v_left_s * R[..., 1]
        + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef
    )
    drag += drag_2[:, 1:] * (
        v_fwd_s * R[..., 0]
        + v_left_s * R[..., 1]
        + v_up_s * R[..., 2] * z_drag_coef
    )

    a_next = act_next + dg - drag

    # 位置和速度更新（Verlet 积分）
    p_next = g_decay(p, GRAD_DECAY ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt ** 2
    v_next = g_decay(v, GRAD_DECAY ** ctl_dt) + (a + a_next) / 2 * ctl_dt

    return act_next, p_next, v_next, a_next


def update_state_vec_pytorch(R, a_thr, v_pred, alpha, yaw_inertia=5.0):
    """
    纯 PyTorch 实现的 update_state_vec，作为参考。
    对应 dynamics_kernel.cu 中的 update_state_vec_cuda_kernel。
    """
    # up vector from thrust
    g_offset = torch.tensor([0.0, 0.0, GRAVITY], device=R.device, dtype=R.dtype)
    a_thr_g = a_thr + g_offset  # a_thr - g_std = a_thr + (0,0,9.80665)
    raw_thrust = a_thr_g.norm(2, -1, keepdim=True)
    # Degenerate case: thrust ≈ 0 → default to upward [0,0,1]
    default_up = torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype).expand_as(a_thr_g)
    up = torch.where(raw_thrust < 1e-8, default_up, a_thr_g / raw_thrust.clamp(min=1e-8))

    # forward vector: blend old forward with v_pred
    fwd_old = R[..., 0]  # (B, 3)
    fwd = fwd_old * yaw_inertia + v_pred  # (B, 3)
    fwd = F.normalize(fwd, 2, -1)
    fwd = (1 - alpha) * fwd + alpha * fwd_old  # (B, 3)

    # Gram-Schmidt: make forward orthogonal to up by adjusting z component
    # fz = (fx*ux + fy*uy) / -uz
    uz_safe = torch.where(up[:, 2:3] >= 0, up[:, 2:3].clamp(min=1e-6), up[:, 2:3].clamp(max=-1e-6))
    fwd_z = (fwd[:, 0:1] * up[:, 0:1] + fwd[:, 1:2] * up[:, 1:2]) / (-uz_safe)
    fwd = torch.cat([fwd[:, 0:1], fwd[:, 1:2], fwd_z], dim=-1)

    # normalize forward
    fwd = F.normalize(fwd, 2, -1)

    # left = cross(up, forward)
    left = torch.cross(up, fwd, dim=-1)

    # R_new columns: [forward, left, up]
    R_new = torch.stack([fwd, left, up], dim=-1)  # (B, 3, 3)
    return R_new
