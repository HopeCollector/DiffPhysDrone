"""
test_cuda_dynamics_backward — 动力学梯度测试 (autograd)
=======================================================

迁移后使用纯 PyTorch 实现，反向传播由 autograd 自动完成。
验证梯度正确流过 run_forward_pytorch 和 g_decay。

覆盖:
  - autograd 梯度有效性
  - grad_decay 正确缩放 d_p 和 d_v
  - 梯度传播路径验证

运行:
    pytest tests/test_cuda_dynamics_backward.py -v
"""

import pytest
import torch

from tests.cuda_test_utils import (
    CTL_DT,
    GRAD_DECAY,
    make_random_rotation,
    run_forward_pytorch,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════

def _make_grad_inputs(B, device="cuda", dtype=torch.float64):
    """生成需要梯度的动力学输入张量"""
    return (
        make_random_rotation(B, device, dtype),                                 # R
        torch.randn((B, 3), dtype=dtype, device=device),                        # dg
        torch.rand((B, 1), dtype=dtype, device=device) + 0.5,                   # z_drag_coef
        torch.rand((B, 2), dtype=dtype, device=device) * 0.3,                   # drag_2
        torch.rand((B, 1), dtype=dtype, device=device) * 10 + 2,               # pitch_ctl_delay
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),    # act_pred
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),    # act
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),    # p
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),    # v
        torch.randn((B, 3), dtype=dtype, device=device) * 0.3,                  # v_wind
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),    # a
    )


# ═══════════════════════════════════════════════════════════
#  梯度有效性
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestBackwardCorrectness:
    """验证 autograd 梯度正确流过 run_forward_pytorch"""

    def _check_gradients(self, B):
        inputs = _make_grad_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        d_out = [torch.randn_like(o) for o in out]
        torch.autograd.backward(out, d_out)

        for name, tensor in [("act_pred", act_pred), ("act", act),
                              ("p", p), ("v", v), ("a", a)]:
            assert tensor.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(tensor.grad).all(), f"{name}.grad contains NaN/Inf"
            assert tensor.grad.abs().sum() > 0, f"{name}.grad is all zeros"

    def test_gradients_B64(self):
        self._check_gradients(64)

    def test_gradients_B1(self):
        self._check_gradients(1)

    def test_gradients_B8(self):
        self._check_gradients(8)


# ═══════════════════════════════════════════════════════════
#  grad_decay 验证
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestBackwardGradDecay:
    """验证 grad_decay 对 d_p 和 d_v 的正确缩放"""

    def test_grad_decay_scales_dp(self):
        """p 的梯度应被 grad_decay^ctl_dt 缩放"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        dg = torch.randn(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.rand(B, 1, device=device, dtype=dtype) + 0.5
        drag_2 = torch.rand(B, 2, device=device, dtype=dtype) * 0.3
        pitch_ctl_delay = torch.rand(B, 1, device=device, dtype=dtype) * 10 + 2
        act_pred = torch.randn(B, 3, device=device, dtype=dtype)
        act = torch.randn(B, 3, device=device, dtype=dtype)
        p = torch.randn(B, 3, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, 3, device=device, dtype=dtype)
        v_wind = torch.randn(B, 3, device=device, dtype=dtype) * 0.3
        a = torch.randn(B, 3, device=device, dtype=dtype)

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        # 只对 p_next 给梯度
        d_p_next = torch.randn(B, 3, device=device, dtype=dtype)
        out[1].backward(d_p_next)

        # d_p = d_p_next * grad_decay^ctl_dt (g_decay 的 backward)
        expected = d_p_next * (GRAD_DECAY ** CTL_DT)
        assert torch.allclose(p.grad, expected, atol=1e-10), (
            f"grad_decay scaling on d_p incorrect, max diff = {(p.grad - expected).abs().max():.2e}"
        )

    def test_grad_decay_scales_dv_zero_drag(self):
        """drag=0 时, v 的梯度基础项应被 grad_decay^ctl_dt 缩放"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        dg = torch.randn(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.rand(B, 1, device=device, dtype=dtype) + 0.5
        drag_2 = torch.zeros(B, 2, device=device, dtype=dtype)  # zero drag
        pitch_ctl_delay = torch.rand(B, 1, device=device, dtype=dtype) * 10 + 2
        act_pred = torch.randn(B, 3, device=device, dtype=dtype)
        act = torch.randn(B, 3, device=device, dtype=dtype)
        p = torch.randn(B, 3, device=device, dtype=dtype)
        v = torch.randn(B, 3, device=device, dtype=dtype, requires_grad=True)
        v_wind = torch.randn(B, 3, device=device, dtype=dtype) * 0.3
        a = torch.randn(B, 3, device=device, dtype=dtype)

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        # 只对 v_next 给梯度
        d_v_next = torch.randn(B, 3, device=device, dtype=dtype)
        out[2].backward(d_v_next)

        # drag=0 → v 的梯度只来自 g_decay + 积分
        expected = d_v_next * (GRAD_DECAY ** CTL_DT)
        assert torch.allclose(v.grad, expected, atol=1e-9), (
            f"grad_decay on d_v (zero drag) incorrect, max diff = {(v.grad - expected).abs().max():.2e}"
        )

    def test_different_grad_decay_values(self):
        """不同 grad_decay 值应产生不同的 d_p 缩放"""
        B = 8
        device, dtype = "cuda", torch.float64
        R = make_random_rotation(B, device, dtype)
        dg = torch.randn(B, 3, device=device, dtype=dtype)
        z_drag_coef = torch.rand(B, 1, device=device, dtype=dtype) + 0.5
        drag_2 = torch.rand(B, 2, device=device, dtype=dtype) * 0.3
        pitch_ctl_delay = torch.rand(B, 1, device=device, dtype=dtype) * 10 + 2
        act_pred = torch.randn(B, 3, device=device, dtype=dtype)
        act = torch.randn(B, 3, device=device, dtype=dtype)
        v = torch.randn(B, 3, device=device, dtype=dtype)
        v_wind = torch.randn(B, 3, device=device, dtype=dtype) * 0.3
        a = torch.randn(B, 3, device=device, dtype=dtype)

        d_p_next = torch.randn(B, 3, device=device, dtype=dtype)

        from bitpilot.env import run_forward, g_decay
        # grad_decay=0.4
        p1 = torch.randn(B, 3, device=device, dtype=dtype, requires_grad=True)
        out1 = run_forward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
                           act_pred, act, p1, v, v_wind, a, 0.4, CTL_DT)
        out1[1].backward(d_p_next)

        # grad_decay=0.8
        p2 = torch.randn(B, 3, device=device, dtype=dtype, requires_grad=True)
        out2 = run_forward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
                           act_pred, act, p2, v, v_wind, a, 0.8, CTL_DT)
        out2[1].backward(d_p_next)

        # grad_decay=0.8 → grad * 0.8^dt vs 0.4^dt → larger
        assert p2.grad.abs().sum() > p1.grad.abs().sum(), (
            "Larger grad_decay should produce larger d_p"
        )


# ═══════════════════════════════════════════════════════════
#  梯度传播路径验证
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestBackwardGradientPaths:
    """验证各输出到各输入的梯度传播路径"""

    def test_a_next_propagates_to_act_pred_and_act(self):
        """a_next 的梯度应传播到 act_pred 和 act"""
        B = 8
        inputs = _make_grad_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        d_a_next = torch.randn_like(out[3])
        out[3].backward(d_a_next)

        assert act_pred.grad.abs().sum() > 0, "a_next gradient should propagate to act_pred"
        assert act.grad.abs().sum() > 0, "a_next gradient should propagate to act"

    def test_v_next_propagates_to_v(self):
        """v_next 的梯度应传播到 v"""
        B = 8
        inputs = _make_grad_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        d_v_next = torch.randn_like(out[2])
        out[2].backward(d_v_next)

        assert v.grad is not None and v.grad.abs().sum() > 0, (
            "v_next gradient should propagate to v"
        )

    def test_zero_upstream_zero_gradient(self):
        """全零上游梯度 → 全零输入梯度"""
        B = 8
        inputs = _make_grad_inputs(B)
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = inputs

        out = run_forward_pytorch(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, CTL_DT,
        )
        zeros = [torch.zeros_like(o) for o in out]
        torch.autograd.backward(out, zeros)

        for name, tensor in [("act_pred", act_pred), ("act", act),
                              ("p", p), ("v", v), ("a", a)]:
            assert torch.allclose(tensor.grad, torch.zeros_like(tensor.grad), atol=1e-15), (
                f"{name}.grad should be zero with zero upstream"
            )
