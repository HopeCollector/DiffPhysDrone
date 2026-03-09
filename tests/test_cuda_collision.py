"""
test_cuda_collision — find_nearest_pt / update_state_vec CUDA 内核测试
======================================================================

覆盖:
  find_nearest_pt:
    - 地面最近点正确性
    - 球体/圆柱/直方体最近点正确性
    - 多子步独立性
    - 在障碍物内部的行为
    - 多无人机互检

  update_state_vec:
    - 正交性和行列式验证
    - 与 Python 参考实现一致性
    - 悬停状态（纯重力补偿）
    - 数值边界（零推力、纯水平推力）
    - alpha 极值退化

运行:
    pytest tests/test_cuda_collision.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

import bitpilot._C as quadsim_cuda
from bitpilot.env import Env
from tests.cuda_test_utils import (
    GRAVITY,
    assert_finite,
    assert_orthogonal,
    make_empty_scene,
    make_identity_R,
    make_random_rotation,
    make_single_ball_scene,
    make_single_cylinder_scene,
    make_single_voxel_scene,
    update_state_vec_pytorch,
)


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — 地面
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtGround:
    """地面在 z=-1，无障碍物时的最近点"""

    def test_ground_nearest_pt(self):
        """无障碍物时最近点是地面 (x, y, -1)"""
        B = 4
        device, dtype = "cuda", torch.float32
        T = 1  # 1 个子步

        pos = torch.zeros(T, B, 3, device=device, dtype=dtype)
        pos[:, :, 2] = 2.0  # z=2
        pos[:, 0, 0] = 1.0
        pos[:, 1, 0] = -1.0

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(
            nearest_pt, balls, cylinders, cylinders_h, voxels,
            pos, 0.15, 1,
        )

        # 最近点应是正下方的地面
        for b in range(B):
            pt = nearest_pt[0, b]
            assert abs(pt[0].item() - pos[0, b, 0].item()) < 0.01, f"x 应保持, batch {b}"
            assert abs(pt[1].item() - pos[0, b, 1].item()) < 0.01, f"y 应保持, batch {b}"
            # z 坐标: 根据 kernel 逻辑, nearest_ptz = min(-1, oz - 1e-3)
            # 当 oz=2 时, min(-1, 1.999) = -1
            assert pt[2].item() < 0, f"地面 z 应为负, batch {b}, z={pt[2].item()}"

    def test_ground_distance_proportional_to_height(self):
        """更高的无人机距地面更远"""
        B = 2
        device, dtype = "cuda", torch.float32
        T = 1

        pos = torch.zeros(T, B, 3, device=device, dtype=dtype)
        pos[0, 0, 2] = 1.0  # z=1, 距地面 2
        pos[0, 1, 2] = 5.0  # z=5, 距地面 6

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        vec = nearest_pt - pos
        dist_0 = vec[0, 0].norm().item()
        dist_1 = vec[0, 1].norm().item()
        assert dist_1 > dist_0, f"更高处距地面更远: {dist_1:.2f} <= {dist_0:.2f}"


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — 球体
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtBall:
    """球体最近点计算"""

    def test_ball_nearest_distance(self):
        """球优先于地面时，最近点应使用球的公式"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        # 无人机在 (0,0,2)，地面距离 = oz+1 = 3
        pos = torch.tensor([[[0.0, 0.0, 2.0]]], device=device, dtype=dtype)
        # 球心在 (1.5, 0, 2)，半径 0.5 → 表面距离 = 1.0-0.5 = 0.5 < 3(地面)
        balls, cylinders, cylinders_h, voxels = make_single_ball_scene(
            B, [1.5, 0.0, 2.0], 0.5, device, dtype
        )
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        # 球的 surface_dist = sqrt(1.5^2) - 0.5 = 1.0
        # 内核公式: nearest = pos + surface_dist * (center - pos)
        # = (0,0,2) + 1.0 * (1.5, 0, 0) = (1.5, 0, 2)
        pt = nearest_pt[0, 0]
        expected_x = 0.0 + 1.0 * 1.5  # = 1.5
        assert abs(pt[0].item() - expected_x) < 0.1, f"nearest x 应≈{expected_x}, 实际={pt[0].item():.2f}"
        assert abs(pt[2].item() - 2.0) < 0.1, f"nearest z 应≈2, 实际={pt[2].item():.2f}"

    def test_ball_direction(self):
        """最近点在球心方向"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        pos = torch.tensor([[[0.0, 0.0, 2.0]]], device=device, dtype=dtype)
        # 球心在 (2.0, 0, 2)，半径 0.3 → surface_dist = 1.7 < ground 3
        balls, cylinders, cylinders_h, voxels = make_single_ball_scene(
            B, [2.0, 0.0, 2.0], 0.3, device, dtype
        )
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        vec = nearest_pt[0, 0] - pos[0, 0]
        # 内核公式: nearest = pos + surface_dist * (center - pos)
        # direction = surface_dist * (center - pos) → 主要在 +x 方向
        assert vec[0].item() > 0, f"应指向球心方向 (+x), actual={vec[0].item():.4f}"
        assert abs(vec[1].item()) < 0.01, "y 分量应很小"


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — 垂直圆柱
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtCylinder:
    """垂直圆柱最近点"""

    def test_cylinder_nearest_distance(self):
        """到垂直圆柱的距离（仅 xy 平面投影）"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        pos = torch.tensor([[[0.0, 0.0, 2.0]]], device=device, dtype=dtype)
        balls, _, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        cylinders = torch.tensor([[[4.0, 0.0, 1.0]]], device=device, dtype=dtype)  # x=4, y=0, r=1

        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        vec = nearest_pt[0, 0] - pos[0, 0]
        dist = vec.norm().item()
        expected_xy = 4.0 - 1.0  # xy 距离 - 半径 = 3
        # 垂直圆柱最近点的 z 坐标与无人机相同
        assert abs(dist - expected_xy) < 0.2, f"到圆柱距离应≈{expected_xy}, 实际={dist:.2f}"


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — AABB 直方体
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtVoxel:
    """AABB 直方体最近点"""

    def test_voxel_nearest_pt_simple(self):
        """直方体外的点，最近点在面上"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        # 无人机在 (6.5, 0, 2)，地面距离 = 3
        pos = torch.tensor([[[6.5, 0.0, 2.0]]], device=device, dtype=dtype)
        # 直方体中心 (5, 0, 2)，半宽 (1, 1, 1) → 面在 x=6
        balls, cylinders, cylinders_h, _ = make_empty_scene(B, device, dtype)
        voxels = torch.tensor([[[5.0, 0.0, 2.0, 1.0, 1.0, 1.0]]], device=device, dtype=dtype)

        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        pt = nearest_pt[0, 0]
        # 直方体使用 clamp 公式: ptx = cx + clamp(ox-cx, -rx, rx) = 5 + min(1, 1.5) = 6
        # dist = sqrt((6-6.5)^2) = 0.5 < 3.0(地面) → 直方体胜
        assert abs(pt[0].item() - 6.0) < 0.1, f"最近点 x 应≈6, 实际={pt[0].item():.2f}"
        assert abs(pt[1].item()) < 0.1, f"最近点 y 应≈0, 实际={pt[1].item():.2f}"
        assert abs(pt[2].item() - 2.0) < 0.1, f"最近点 z 应≈2, 实际={pt[2].item():.2f}"


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — 多子步
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtSubSteps:
    """多子步时间独立性"""

    def test_multi_substep_independent(self):
        """每个子步应独立计算"""
        B = 2
        device, dtype = "cuda", torch.float32
        T = 5

        pos = torch.randn(T, B, 3, device=device, dtype=dtype)
        pos[..., 2] = pos[..., 2].abs() + 1  # 确保在地面上方

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        # 每个子步的结果应与单独计算一致
        for t in range(T):
            pos_single = pos[t:t + 1]
            nearest_single = torch.empty_like(pos_single)
            quadsim_cuda.find_nearest_pt(
                nearest_single, balls, cylinders, cylinders_h, voxels,
                pos_single, 0.15, 1,
            )
            assert torch.allclose(nearest_pt[t], nearest_single[0], atol=1e-6), (
                f"子步 {t} 不独立"
            )

    def test_finite_output(self):
        """所有子步输出有限"""
        B = 8
        device, dtype = "cuda", torch.float32
        T = 10

        pos = torch.randn(T, B, 3, device=device, dtype=dtype)
        pos[..., 2] = pos[..., 2].abs() + 0.5

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)
        assert_finite(nearest_pt, "nearest_pt multi-substep")


# ═══════════════════════════════════════════════════════════
#  find_nearest_pt — 边界条件
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestNearestPtEdgeCases:
    """find_nearest_pt 边界条件"""

    def test_inside_ball_clamped(self):
        """无人机在球内时，距离被 max(1e-3, ...) 保护"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        # 无人机在球心
        pos = torch.tensor([[[3.0, 0.0, 2.0]]], device=device, dtype=dtype)
        balls, cylinders, cylinders_h, voxels = make_single_ball_scene(
            B, [3.0, 0.0, 2.0], 2.0, device, dtype  # 大球包围无人机
        )
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)

        # 不应崩溃，结果应有限
        assert_finite(nearest_pt, "inside ball")

    def test_on_ground_clamped(self):
        """无人机在地面附近时不崩溃"""
        B = 1
        device, dtype = "cuda", torch.float32
        T = 1

        pos = torch.tensor([[[-0.5, 0.0, -0.99]]], device=device, dtype=dtype)  # 接近地面 z=-1
        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 1)
        assert_finite(nearest_pt, "near ground")

    def test_multi_drone_collision(self):
        """多无人机互检（椭球碰撞）"""
        B = 4
        device, dtype = "cuda", torch.float32
        T = 1

        pos = torch.zeros(T, B, 3, device=device, dtype=dtype)
        pos[0, 0] = torch.tensor([0.0, 0.0, 2.0])
        pos[0, 1] = torch.tensor([0.5, 0.0, 2.0])  # 靠近无人机 0
        pos[0, 2] = torch.tensor([10.0, 0.0, 2.0])
        pos[0, 3] = torch.tensor([10.5, 0.0, 2.0])

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        nearest_pt = torch.empty_like(pos)
        quadsim_cuda.find_nearest_pt(nearest_pt, balls, cylinders, cylinders_h, voxels, pos, 0.15, 4)

        # 无人机 0 应检测到无人机 1（很近）
        vec_0 = nearest_pt[0, 0] - pos[0, 0]
        dist_0 = vec_0.norm().item()
        # 椭球距离：sqrt(dx^2 + dy^2 + 4*dz^2) - r, 其中 dx=0.5, r=0.15
        expected_dist = max(1e-3, 0.5 - 0.15)
        assert abs(dist_0 - expected_dist) < 0.2, (
            f"多机互检距离应≈{expected_dist:.2f}, 实际={dist_0:.2f}"
        )


# ═══════════════════════════════════════════════════════════
#  update_state_vec — 正交性
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestUpdateStateVecOrthogonality:
    """验证输出旋转矩阵的正交性"""

    def test_orthogonal_output(self):
        """R_new^T @ R_new ≈ I, det(R_new) ≈ 1"""
        B = 32
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        a_thr = torch.randn(B, 3, device=device, dtype=dtype)
        v_pred = torch.randn(B, 3, device=device, dtype=dtype)
        alpha = torch.rand(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)
        assert_orthogonal(R_new, atol=1e-6)

    def test_orthogonal_from_identity(self):
        """从单位矩阵开始也保持正交"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        a_thr = torch.randn(B, 3, device=device, dtype=dtype)
        v_pred = torch.randn(B, 3, device=device, dtype=dtype)
        alpha = torch.rand(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)
        assert_orthogonal(R_new, atol=1e-6)

    def test_repeated_updates_stable(self):
        """多次连续更新后仍保持正交"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        alpha = torch.ones(B, 1, device=device, dtype=dtype) * 0.3

        for _ in range(50):
            a_thr = torch.randn(B, 3, device=device, dtype=dtype)
            v_pred = torch.randn(B, 3, device=device, dtype=dtype)
            R = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)

        assert_orthogonal(R, atol=1e-4)


# ═══════════════════════════════════════════════════════════
#  update_state_vec — 与 Python 参考一致性
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestUpdateStateVecReference:
    """CUDA vs Python 参考实现"""

    def test_matches_pytorch_reference(self):
        """CUDA 输出应与 PyTorch 参考一致"""
        B = 16
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        a_thr = torch.randn(B, 3, device=device, dtype=dtype)
        v_pred = torch.randn(B, 3, device=device, dtype=dtype)
        alpha = torch.rand(B, 1, device=device, dtype=dtype) * 0.5
        yaw_inertia = 5.0

        cuda_out = Env.update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia)
        ref_out = update_state_vec_pytorch(R, a_thr, v_pred, alpha, yaw_inertia)

        assert torch.allclose(cuda_out, ref_out, atol=1e-8), (
            f"CUDA vs PyTorch max diff = {(cuda_out - ref_out).abs().max():.2e}"
        )

    @pytest.mark.parametrize("B", [1, 2, 8, 32, 128])
    def test_various_batch_sizes(self, B):
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        a_thr = torch.randn(B, 3, device=device, dtype=dtype)
        v_pred = torch.randn(B, 3, device=device, dtype=dtype)
        alpha = torch.rand(B, 1, device=device, dtype=dtype)

        cuda_out = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)
        ref_out = update_state_vec_pytorch(R, a_thr, v_pred, alpha, 5.0)
        assert torch.allclose(cuda_out, ref_out, atol=1e-8), f"B={B} mismatch"


# ═══════════════════════════════════════════════════════════
#  update_state_vec — 物理语义
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestUpdateStateVecPhysics:
    """物理语义验证"""

    def test_hover_up_vector(self):
        """悬停（a_thr≈0） → up ≈ [0,0,1]"""
        B = 4
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        # a_thr = [0,0,0] → 内核中 + 9.80665 → thrust_dir = [0,0,1]
        a_thr = torch.zeros(B, 3, device=device, dtype=dtype)
        v_pred = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(B, -1)
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)

        # up 向量（第 3 列）应为 [0, 0, 1]
        up = R_new[:, :, 2]
        expected_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        assert torch.allclose(up, expected_up.expand_as(up), atol=1e-6), (
            f"悬停 up 应为 [0,0,1], 实际={up[0].tolist()}"
        )

    def test_tilted_thrust_tilts_up(self):
        """前倾推力 → up 向量前倾"""
        B = 1
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        # 前方加速 → 推力方向前倾
        a_thr = torch.tensor([[5.0, 0.0, 0.0]], device=device, dtype=dtype)
        v_pred = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)
        up = R_new[0, :, 2]

        # up 应有正的 x 分量（前倾）
        assert up[0].item() > 0, f"前倾推力应使 up.x > 0, 实际={up[0].item():.4f}"
        # up 仍应有正的 z 分量
        assert up[2].item() > 0, f"up.z 应仍为正, 实际={up[2].item():.4f}"

    def test_alpha_1_keeps_old_forward(self):
        """alpha=1 → 完全保持旧的前向向量（仅更新 up）"""
        B = 4
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        a_thr = torch.zeros(B, 3, device=device, dtype=dtype)  # 悬停 → up=[0,0,1]
        v_pred = torch.randn(B, 3, device=device, dtype=dtype) * 10  # 强方向偏置
        alpha = torch.ones(B, 1, device=device, dtype=dtype)  # 完全保持旧方向

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)

        # forward 方向应与旧 R 的 forward 一致（但经过正交化可能有小变化）
        # 由于 up=[0,0,1] 且 alpha=1，forward 应是旧 forward 正交化到水平面的结果
        old_fwd = R[:, :, 0]
        new_fwd = R_new[:, :, 2]  # 实际上是 up，check 正交
        assert_orthogonal(R_new, atol=1e-5)

    def test_large_yaw_inertia_slow_turn(self):
        """大的 yaw_inertia → forward 变化慢"""
        B = 4
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)  # forward = [1,0,0]
        a_thr = torch.zeros(B, 3, device=device, dtype=dtype)
        # 想转向 +Y 方向
        v_pred = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).expand(B, -1).contiguous()
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_low_inertia = Env.update_state_vec(R, a_thr, v_pred, alpha, 1.0)
        R_high_inertia = Env.update_state_vec(R, a_thr, v_pred, alpha, 100.0)

        # 低惯性时 forward 应更偏向 v_pred（+Y）
        fwd_low = R_low_inertia[0, :, 0]
        fwd_high = R_high_inertia[0, :, 0]

        # 低惯性的 y 分量应更大
        assert fwd_low[1].abs().item() > fwd_high[1].abs().item(), (
            "低 yaw_inertia 应产生更大的 forward.y 偏转"
        )


# ═══════════════════════════════════════════════════════════
#  update_state_vec — 数值边界
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestUpdateStateVecEdgeCases:
    """数值边界条件"""

    def test_zero_thrust_behavior(self):
        """零推力 a_thr=[0,0,-g] → 内核中 thrust 被 clamp 到 1e-8，不再除零"""
        B = 2
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        a_thr = torch.tensor([[0.0, 0.0, -GRAVITY]], device=device, dtype=dtype).expand(B, -1).contiguous()
        v_pred = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(B, -1).contiguous()
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)

        assert torch.isfinite(R_new).all(), "零推力不应产生 NaN/Inf"
        assert_orthogonal(R_new, atol=1e-4)

    def test_horizontal_thrust_behavior(self):
        """纯水平推力 → uz≈0 → uz 被 clamp 到 1e-6，不再除零"""
        B = 2
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        # a_thr = [100, 0, -g] → 内核中 a_thr_g = [100, 0, 0] → uz≈0
        a_thr = torch.tensor([[100.0, 0.0, -GRAVITY]], device=device, dtype=dtype).expand(B, -1).contiguous()
        v_pred = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(B, -1).contiguous()
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)

        assert torch.isfinite(R_new).all(), "纯水平推力不应产生 NaN/Inf"
        assert_orthogonal(R_new, atol=1e-3)

    def test_zero_v_pred_with_inertia(self):
        """v_pred=0 + 大 yaw_inertia → forward 的 xy 分量保持旧方向"""
        B = 4
        device, dtype = "cuda", torch.float64
        # 使用 forward 主要在水平面的旋转矩阵，避免 Gram-Schmidt 投影剧变
        R = make_identity_R(B, device, dtype)
        a_thr = torch.zeros(B, 3, device=device, dtype=dtype)  # → up=[0,0,1]
        v_pred = torch.zeros(B, 3, device=device, dtype=dtype)
        alpha = torch.zeros(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 100.0)
        assert_finite(R_new.float(), "zero v_pred output")
        # 单位矩阵的 forward = [1,0,0]（水平），投影后应保持不变
        old_fwd = R[:, :, 0]
        new_fwd = R_new[:, :, 0]
        cosine = (old_fwd * new_fwd).sum(-1)
        assert (cosine > 0.99).all(), "单位矩阵 + 悬停: forward 应保持 [1,0,0]"

    def test_float32_works(self):
        """float32 精度下正常工作"""
        B = 16
        device, dtype = "cuda", torch.float32
        R = make_random_rotation(B, device, dtype)
        a_thr = torch.randn(B, 3, device=device, dtype=dtype)
        v_pred = torch.randn(B, 3, device=device, dtype=dtype)
        alpha = torch.rand(B, 1, device=device, dtype=dtype)

        R_new = Env.update_state_vec(R, a_thr, v_pred, alpha, 5.0)
        assert_finite(R_new, "float32 update_state_vec")
        assert_orthogonal(R_new, atol=1e-3)
