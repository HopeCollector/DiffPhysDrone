"""
test_cuda_render — render / rerender_backward CUDA 内核测试
=============================================================

覆盖:
  - 地面渲染正确性
  - 球体/圆柱/直方体射线追踪正确性
  - 多无人机互相遮挡
  - 空场景默认深度
  - FOV 一致性
  - rerender_backward 法向量计算
  - 数值边界条件

运行:
    pytest tests/test_cuda_render.py -v
"""

import math

import pytest
import torch

import bitpilot._C as quadsim_cuda
from tests.cuda_test_utils import (
    assert_finite,
    make_empty_scene,
    make_identity_R,
    make_single_ball_scene,
    make_single_cylinder_scene,
    make_single_voxel_scene,
)


# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

H, W = 48, 64
FOV_X_HALF_TAN = 0.82
DRONE_RADIUS = 0.15


def _render(B, R, pos, balls, cylinders, cylinders_h, voxels,
            drone_radius=DRONE_RADIUS, n_drones_per_group=1,
            fov=FOV_X_HALF_TAN, dtype=torch.float32):
    """便捷渲染封装"""
    device = pos.device
    canvas = torch.empty(B, H, W, device=device, dtype=dtype)
    flow = torch.empty(B, 0, H, W, device=device, dtype=dtype)
    R_old = R.clone()
    pos_old = pos.clone()
    quadsim_cuda.render(
        canvas, flow, balls, cylinders, cylinders_h, voxels,
        R, R_old, pos, pos_old,
        float(drone_radius), int(n_drones_per_group), float(fov),
    )
    return canvas


# ═══════════════════════════════════════════════════════════
#  地面渲染
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderGround:
    """地面在 z=-1 是一个无限平面"""

    def test_ground_depth_looking_down(self):
        """
        相机在 z=2 朝正下方 → 中心像素深度 ≈ 3 (从 z=2 到 z=-1)。
        
        R 矩阵需要使射线正下方：
        - 射线方向公式: d = R[:,0] - fu*R[:,2] - fv*R[:,1]
        - 中心像素 fu≈0, fv≈0 → d ≈ R[:,0]
        - 要射线朝下: R[:,0] = (0,0,-1)
        - 为保持正交: R = [[0,0,-1],[0,1,0],[1,0,0]]^T → columns = [0,0,-1], [0,1,0], [1,0,0]
        """
        B = 1
        device, dtype = "cuda", torch.float32
        # R 的列: col0=[0,0,-1](前/射线方向), col1=[0,1,0](左), col2=[1,0,0](上)
        R = torch.tensor([
            [[0, 0, 1],
             [0, 1, 0],
             [-1, 0, 0]],
        ], device=device, dtype=dtype)
        pos = torch.tensor([[0, 0, 2]], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        # 中心像素（约 H/2, W/2）
        center_depth = canvas[0, H // 2, W // 2].item()
        # 射线从 z=2 向下，与地面 z=-1 交点距离为 3
        assert abs(center_depth - 3.0) < 0.5, (
            f"地面深度应≈3.0, 实际={center_depth:.2f}"
        )

    def test_ground_higher_altitude_deeper(self):
        """更高的 z → 更大的地面深度"""
        device, dtype = "cuda", torch.float32
        B = 2
        R = torch.tensor([
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        ], device=device, dtype=dtype)
        pos = torch.tensor([[0, 0, 2], [0, 0, 5]], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        depth_low = canvas[0, H // 2, W // 2].item()
        depth_high = canvas[1, H // 2, W // 2].item()
        assert depth_high > depth_low, f"更高处深度应更大: {depth_high:.2f} <= {depth_low:.2f}"


# ═══════════════════════════════════════════════════════════
#  球体渲染
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderBall:
    """球体射线-球交叉检测"""

    def test_ball_in_front(self):
        """正前方的球应遮挡部分像素"""
        B = 1
        device, dtype = "cuda", torch.float32
        # 相机朝 +X, 没有倾斜
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        # 球心在 (5, 0, 2)，半径 1
        balls, cylinders, cylinders_h, voxels = make_single_ball_scene(
            B, [5.0, 0.0, 2.0], 1.0, device, dtype
        )
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center_depth = canvas[0, H // 2, W // 2].item()
        # 从 x=0 到球面最近点 x=5-1=4
        assert center_depth < 5.0, f"球遮挡应使深度 < 5, 实际={center_depth:.2f}"
        assert center_depth > 3.0, f"球交叉深度应 > 3 (≈4), 实际={center_depth:.2f}"

    def test_ball_behind_invisible(self):
        """相机背后的球不应可见"""
        B = 1
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)  # 朝 +X
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        # 球在 x=-5（相机背后）
        balls, cylinders, cylinders_h, voxels = make_single_ball_scene(
            B, [-5.0, 0.0, 2.0], 1.0, device, dtype
        )
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center_depth = canvas[0, H // 2, W // 2].item()
        # 应该看到地面或远距离
        assert center_depth >= 3.0, f"背后的球不应遮挡, depth={center_depth:.2f}"


# ═══════════════════════════════════════════════════════════
#  垂直圆柱渲染
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderCylinder:
    """垂直和水平圆柱交叉"""

    def test_vertical_cylinder_in_front(self):
        """正前方的垂直圆柱"""
        B = 1
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        # 垂直圆柱在 (5, 0), r=1
        balls, _, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        cylinders = torch.tensor([[[5.0, 0.0, 1.0]]], device=device, dtype=dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center_depth = canvas[0, H // 2, W // 2].item()
        assert center_depth < 5.0, f"圆柱遮挡应使深度 < 5, 实际={center_depth:.2f}"
        assert center_depth > 3.0, f"圆柱交叉深度应 > 3, 实际={center_depth:.2f}"


# ═══════════════════════════════════════════════════════════
#  AABB 直方体渲染
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderVoxel:
    """AABB 直方体射线交叉"""

    def test_voxel_in_front(self):
        """正前方的直方体"""
        B = 1
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        # 直方体在 (5, 0, 2), 半宽 (1, 1, 1)
        balls, cylinders, cylinders_h, _ = make_empty_scene(B, device, dtype)
        voxels = torch.tensor([[[5.0, 0.0, 2.0, 1.0, 1.0, 1.0]]], device=device, dtype=dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center_depth = canvas[0, H // 2, W // 2].item()
        # 从 x=0 到 AABB 近面 x=5-1=4
        assert abs(center_depth - 4.0) < 0.5, f"直方体近面深度应≈4, 实际={center_depth:.2f}"

    def test_voxel_behind_invisible(self):
        """背后的直方体不可见"""
        B = 1
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, _ = make_empty_scene(B, device, dtype)
        voxels = torch.tensor([[[-5.0, 0.0, 2.0, 1.0, 1.0, 1.0]]], device=device, dtype=dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center_depth = canvas[0, H // 2, W // 2].item()
        assert center_depth >= 3.0, f"背后直方体不应遮挡, depth={center_depth:.2f}"


# ═══════════════════════════════════════════════════════════
#  多无人机遮挡
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderMultiDrone:
    """多无人机渲染（n_drones_per_group > 1）"""

    def test_other_drone_visible(self):
        """同组另一架无人机在正前方应可见"""
        B = 2  # 两架无人机为一组
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        # 无人机 0 在原点，无人机 1 在 (3, 0, 2)
        pos = torch.tensor([
            [0.0, 0.0, 2.0],  # 我
            [3.0, 0.0, 2.0],  # 前方的无人机
        ], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(
            B, R, pos, balls, cylinders, cylinders_h, voxels,
            drone_radius=0.15, n_drones_per_group=2,
        )

        # 无人机 0 看无人机 1（椭球体，r=0.15）
        center_depth_0 = canvas[0, H // 2, W // 2].item()
        # 距离约 3 - 0.15 = 2.85（椭球可能不精确）
        assert center_depth_0 < 3.5, f"应看到另一架无人机, depth={center_depth_0:.2f}"

    def test_different_group_invisible(self):
        """不同组的无人机互不可见"""
        B = 2
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
        ], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        # 每组 1 架 → 互不可见
        canvas = _render(
            B, R, pos, balls, cylinders, cylinders_h, voxels,
            drone_radius=0.15, n_drones_per_group=1,
        )

        center_depth_0 = canvas[0, H // 2, W // 2].item()
        # 看不到另一架，应只看到地面（≈3）或远处（100）
        assert center_depth_0 >= 2.9, f"不同组的无人机应不可见, depth={center_depth_0:.2f}"


# ═══════════════════════════════════════════════════════════
#  空场景和默认值
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderEmptyScene:
    """空场景（朝上看，无障碍物）"""

    def test_empty_scene_max_dist(self):
        """朝上看时空场景深度应为 100（min_dist 初始值）"""
        B = 1
        device, dtype = "cuda", torch.float32
        # 朝上: col0=[0,0,1], col1=[0,1,0], col2=[-1,0,0]
        R = torch.tensor([
            [[0, 0, -1],
             [0, 1, 0],
             [1, 0, 0]],
        ], device=device, dtype=dtype)
        pos = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)

        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)

        center = canvas[0, H // 2, W // 2].item()
        assert center == 100.0, f"空场景朝上看应为 100, 实际={center:.2f}"


# ═══════════════════════════════════════════════════════════
#  渲染基本性质
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRenderProperties:
    """渲染结果的基本性质"""

    def test_output_shape(self):
        B = 4
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.zeros(B, 3, device=device, dtype=dtype)
        pos[:, 2] = 2.0
        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)
        assert canvas.shape == (B, H, W)

    def test_all_positive(self):
        """深度值应全部 > 0"""
        B = 4
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.zeros(B, 3, device=device, dtype=dtype)
        pos[:, 2] = 2.0
        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)
        assert (canvas > 0).all(), "深度应全为正值"

    def test_no_nan(self):
        """渲染结果不应包含 NaN"""
        B = 8
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.randn(B, 3, device=device, dtype=dtype)
        pos[:, 2] = 2.0 + torch.rand(B, device=device)
        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)
        assert_finite(canvas, "render canvas")

    def test_closer_object_smaller_depth(self):
        """更近的障碍物应产生更小的深度"""
        B = 2
        device, dtype = "cuda", torch.float32
        R = make_identity_R(B, device, dtype)
        pos = torch.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ], device=device, dtype=dtype)

        # batch 0: 球在 x=3, batch 1: 球在 x=8
        balls, cylinders, cylinders_h, voxels = make_empty_scene(B, device, dtype)
        balls[0, 0] = torch.tensor([3.0, 0.0, 2.0, 0.5], device=device, dtype=dtype)
        balls[1, 0] = torch.tensor([8.0, 0.0, 2.0, 0.5], device=device, dtype=dtype)

        canvas = _render(B, R, pos, balls, cylinders, cylinders_h, voxels)
        d0 = canvas[0, H // 2, W // 2].item()
        d1 = canvas[1, H // 2, W // 2].item()
        assert d0 < d1, f"近球应有更小深度: {d0:.2f} >= {d1:.2f}"


# ═══════════════════════════════════════════════════════════
#  rerender_backward（法向量计算）
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestRerenderBackward:
    """rerender_backward_cuda 法向量测试"""

    def test_output_shape(self):
        """验证输出形状 (B, 3, H, W)"""
        B, H_out, W_out = 2, 12, 16
        device, dtype = "cuda", torch.float32
        # 输入: (B, 1, 2*H, 2*W)
        depth = torch.ones(B, 1, 2 * H_out, 2 * W_out, device=device, dtype=dtype) * 5.0
        dddp = torch.empty(B, 3, H_out, W_out, device=device, dtype=dtype)
        quadsim_cuda.rerender_backward(depth, dddp, FOV_X_HALF_TAN)
        assert dddp.shape == (B, 3, H_out, W_out)

    def test_flat_surface_normal(self):
        """平面深度图 → 法向量应接近 (-1, 0, 0)"""
        B, H_out, W_out = 1, 12, 16
        device, dtype = "cuda", torch.float32
        # 均匀深度
        depth = torch.ones(B, 1, 2 * H_out, 2 * W_out, device=device, dtype=dtype) * 5.0
        dddp = torch.empty(B, 3, H_out, W_out, device=device, dtype=dtype)
        quadsim_cuda.rerender_backward(depth, dddp, FOV_X_HALF_TAN)

        # 均匀深度 → dddy = dddz = 0 → 法向量 ∝ (-1, 0, 0)
        # 但被 clamp 到范数 8 → (-1/8, 0, 0)
        assert_finite(dddp, "rerender_backward output")
        # 第一通道应主要为负（指向相机方向）
        assert (dddp[0, 0] < 0).all(), "平面法向量第一分量应为负"
        # 第二、三通道应约为 0
        assert dddp[0, 1].abs().max() < 0.01, "平面法向量 y 分量应≈0"
        assert dddp[0, 2].abs().max() < 0.01, "平面法向量 z 分量应≈0"

    def test_finite_with_varying_depth(self):
        """不同深度的区域不产生 NaN"""
        B, H_out, W_out = 2, 12, 16
        device, dtype = "cuda", torch.float32
        depth = torch.rand(B, 1, 2 * H_out, 2 * W_out, device=device, dtype=dtype) * 10 + 0.5
        dddp = torch.empty(B, 3, H_out, W_out, device=device, dtype=dtype)
        quadsim_cuda.rerender_backward(depth, dddp, FOV_X_HALF_TAN)
        assert_finite(dddp, "rerender_backward varying depth")

    def test_normalization_bound(self):
        """法向量范数被 clamp 到最大 1/8"""
        B, H_out, W_out = 1, 12, 16
        device, dtype = "cuda", torch.float32
        depth = torch.ones(B, 1, 2 * H_out, 2 * W_out, device=device, dtype=dtype) * 5
        dddp = torch.empty(B, 3, H_out, W_out, device=device, dtype=dtype)
        quadsim_cuda.rerender_backward(depth, dddp, FOV_X_HALF_TAN)

        # norm(dddp) 在每个像素应 ≤ 1 (因为 norm_div >= 1)
        norms = dddp.norm(2, dim=1)  # (B, H, W)
        assert (norms <= 1.0 + 1e-5).all(), f"法向量范数应 ≤ 1, max={norms.max():.4f}"
