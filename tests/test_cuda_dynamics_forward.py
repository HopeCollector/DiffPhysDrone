"""
test_cuda_dynamics_forward — run_forward 动力学前向传播全面测试
=============================================================

迁移后使用纯 PyTorch 实现，不再依赖 CUDA run_forward 内核。

覆盖:
  - 物理不变量（零速度、零时间步、delay 极值）
  - 批量边界 (B=1, B=2, B=256)
  - 数值边界条件（极大速度、零推力）

运行:
    pytest tests/test_cuda_dynamics_forward.py -v
"""

import pytest
import torch

from tests.cuda_test_utils import (
    CTL_DT,
    GRAVITY,
    make_identity_R,
    make_random_rotation,
    run_forward_pytorch,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════

def _make_dynamics_inputs(B, device="cuda", dtype=torch.float64):
    """生成随机动力学输入张量"""
    return (
        make_random_rotation(B, device, dtype),                      # R
        torch.randn((B, 3), dtype=dtype, device=device),             # dg
        torch.rand((B, 1), dtype=dtype, device=device) + 0.5,       # z_drag_coef (正值)
        torch.rand((B, 2), dtype=dtype, device=device) * 0.3,       # drag_2 (正值小量)
        torch.rand((B, 1), dtype=dtype, device=device) * 10 + 2,    # pitch_ctl_delay (2~12)
        torch.randn((B, 3), dtype=dtype, device=device),             # act_pred
        torch.randn((B, 3), dtype=dtype, device=device),             # act
        torch.randn((B, 3), dtype=dtype, device=device),             # p
        torch.randn((B, 3), dtype=dtype, device=device),             # v
        torch.randn((B, 3), dtype=dtype, device=device) * 0.3,      # v_wind
        torch.randn((B, 3), dtype=dtype, device=device),             # a
    )


# ═══════════════════════════════════════════════════════════
#  物理不变量
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestForwardPhysicsInvariants:
    """验证动力学方程的物理不变量"""

    def test_zero_velocity_zero_drag(self):
        """零速度 + 零风速 → 无阻力，a_next = act_next + dg"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        dg = torch.randn(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.ones(B, 1, device=device, dtype=dtype)
        drag_2 = torch.ones(B, 2, device=device, dtype=dtype) * 0.5
        pitch_ctl_delay = torch.ones(B, 1, device=device, dtype=dtype) * 10
        act_pred = torch.randn(B, 3, device=device, dtype=dtype)
        act = torch.randn(B, 3, device=device, dtype=dtype)
        p = torch.randn(B, 3, device=device, dtype=dtype)
        v = torch.zeros(B, 3, device=device, dtype=dtype)
        v_wind = torch.zeros(B, 3, device=device, dtype=dtype)
        a = torch.randn(B, 3, device=device, dtype=dtype)

        act_next, p_next, v_next, a_next = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        expected_a_next = act_next + dg
        assert torch.allclose(a_next, expected_a_next, atol=1e-10), (
            f"零速度时 a_next 不等于 act_next+dg, diff={(a_next - expected_a_next).abs().max():.2e}"
        )

    def test_ctl_dt_zero_identity(self):
        """ctl_dt=0 → act_next=act, p_next=p, v_next=v"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs
        act_next, p_next, v_next, a_next = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, 0.0,
        )
        assert torch.allclose(act_next, act, atol=1e-10), "ctl_dt=0: act_next should equal act"
        assert torch.allclose(p_next, p, atol=1e-10), "ctl_dt=0: p_next should equal p"
        assert torch.allclose(v_next, v, atol=1e-10), "ctl_dt=0: v_next should equal v"

    def test_large_pitch_ctl_delay_follows_pred(self):
        """极大 pitch_ctl_delay → alpha≈0 → act_next ≈ act_pred"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, _, act_pred, act, p, v, v_wind, a = inputs
        pitch_ctl_delay = torch.full((B, 1), 1000.0, device="cuda", dtype=torch.float64)
        act_next, _, _, _ = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        assert torch.allclose(act_next, act_pred, atol=1e-6), (
            "极大 pitch_ctl_delay: act_next should ≈ act_pred"
        )

    def test_zero_pitch_ctl_delay_keeps_act(self):
        """pitch_ctl_delay=0 → alpha=1 → act_next = act"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, _, act_pred, act, p, v, v_wind, a = inputs
        pitch_ctl_delay = torch.zeros(B, 1, device="cuda", dtype=torch.float64)
        act_next, _, _, _ = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        assert torch.allclose(act_next, act, atol=1e-10), (
            "pitch_ctl_delay=0: act_next should equal act"
        )

    def test_p_next_only_depends_on_old_state(self):
        """p_next 不依赖 act_pred（只依赖 p, v, a）"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, _, act, p, v, v_wind, a = inputs
        act_pred_1 = torch.randn(B, 3, device="cuda", dtype=torch.float64)
        act_pred_2 = torch.randn(B, 3, device="cuda", dtype=torch.float64)

        _, p1, _, _ = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred_1, act, p, v, v_wind, a, CTL_DT,
        )
        _, p2, _, _ = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred_2, act, p, v, v_wind, a, CTL_DT,
        )
        assert torch.allclose(p1, p2, atol=1e-12), "p_next should not depend on act_pred"

    def test_verlet_integration(self):
        """验证 Verlet 积分公式: p_next = p + v*dt + 0.5*a*dt^2"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs
        _, p_next, _, _ = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        expected_p = p + v * CTL_DT + 0.5 * a * CTL_DT ** 2
        assert torch.allclose(p_next, expected_p, atol=1e-10), "Verlet integration formula mismatch"

    def test_velocity_update_formula(self):
        """验证速度更新: v_next = v + 0.5*(a + a_next)*dt"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs
        _, _, v_next, a_next = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        expected_v = v + 0.5 * (a + a_next) * CTL_DT
        assert torch.allclose(v_next, expected_v, atol=1e-10), "Velocity update formula mismatch"


# ═══════════════════════════════════════════════════════════
#  批量大小边界
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestForwardBatchSizes:

    @pytest.mark.parametrize("B", [1, 2, 3, 7, 16, 128, 256])
    def test_various_batch_sizes(self, B):
        """不同批量大小均能正确运行且输出有限"""
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs
        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        names = ["act_next", "p_next", "v_next", "a_next"]
        for i, name in enumerate(names):
            assert out[i].shape == (B, 3), f"B={B}: {name} wrong shape"
            assert torch.isfinite(out[i]).all(), f"B={B}: {name} contains NaN/Inf"


# ═══════════════════════════════════════════════════════════
#  数值边界条件
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestForwardNumericalEdgeCases:

    def test_large_velocity_finite(self):
        """极大速度 (v~100) 不产生 NaN/Inf"""
        B = 8
        inputs = _make_dynamics_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, _, v_wind, a = inputs
        v = torch.ones(B, 3, device="cuda", dtype=torch.float64) * 100.0
        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        for i, name in enumerate(["act_next", "p_next", "v_next", "a_next"]):
            assert torch.isfinite(out[i]).all(), f"极大速度: {name} 包含 NaN/Inf"

    def test_zero_thrust_finite(self):
        """零推力 (act=[0,0,-g]) 不产生 NaN/Inf"""
        B = 4
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        dg = torch.zeros(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.ones(B, 1, device=device, dtype=dtype)
        drag_2 = torch.zeros(B, 2, device=device, dtype=dtype)
        pitch_ctl_delay = torch.ones(B, 1, device=device, dtype=dtype) * 10
        act_pred = torch.zeros(B, 3, device=device, dtype=dtype)
        act_pred[:, 2] = -GRAVITY
        act = act_pred.clone()
        p = torch.zeros(B, 3, device=device, dtype=dtype)
        v = torch.zeros(B, 3, device=device, dtype=dtype)
        v_wind = torch.zeros(B, 3, device=device, dtype=dtype)
        a = torch.zeros(B, 3, device=device, dtype=dtype)

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        for i, name in enumerate(["act_next", "p_next", "v_next", "a_next"]):
            assert torch.isfinite(out[i]).all(), f"零推力: {name} 包含 NaN/Inf"

    def test_wind_equals_velocity_zero_drag(self):
        """v_wind == v → 相对风速为零 → 阻力为零"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        dg = torch.randn(B, 3, device=device, dtype=dtype) * 0.1
        z_drag_coef = torch.ones(B, 1, device=device, dtype=dtype) * 2
        drag_2 = torch.ones(B, 2, device=device, dtype=dtype)
        pitch_ctl_delay = torch.ones(B, 1, device=device, dtype=dtype) * 10
        act_pred = torch.randn(B, 3, device=device, dtype=dtype)
        act = torch.randn(B, 3, device=device, dtype=dtype)
        p = torch.randn(B, 3, device=device, dtype=dtype)
        v = torch.randn(B, 3, device=device, dtype=dtype) * 5
        v_wind = v.clone()
        a = torch.randn(B, 3, device=device, dtype=dtype)

        act_next, _, _, a_next = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        expected = act_next + dg
        assert torch.allclose(a_next, expected, atol=1e-9), (
            f"v_wind==v: drag should be zero, max diff = {(a_next - expected).abs().max():.2e}"
        )

    def test_drag_symmetry(self):
        """反向速度应产生反向阻力"""
        B = 4
        device, dtype = "cuda", torch.float64
        R = make_identity_R(B, device, dtype)
        dg = torch.zeros(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.ones(B, 1, device=device, dtype=dtype)
        drag_2 = torch.tensor([[0.3, 0.0]], device=device, dtype=dtype).expand(B, -1).contiguous()
        pitch_ctl_delay = torch.ones(B, 1, device=device, dtype=dtype) * 10
        act_pred = torch.zeros(B, 3, device=device, dtype=dtype)
        act = torch.zeros(B, 3, device=device, dtype=dtype)
        p = torch.zeros(B, 3, device=device, dtype=dtype)
        v_wind = torch.zeros(B, 3, device=device, dtype=dtype)
        a = torch.zeros(B, 3, device=device, dtype=dtype)

        v_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        v_pos[:, 0] = 5.0
        _, _, _, a_next_pos = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v_pos, v_wind, a, CTL_DT,
        )

        v_neg = torch.zeros(B, 3, device=device, dtype=dtype)
        v_neg[:, 0] = -5.0
        _, _, _, a_next_neg = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v_neg, v_wind, a, CTL_DT,
        )

        assert torch.allclose(a_next_pos, -a_next_neg, atol=1e-10), (
            "Drag should be antisymmetric with velocity direction"
        )
