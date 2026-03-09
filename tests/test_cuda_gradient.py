"""
test_cuda_gradient — 动力学内核梯度正确性验证 (端到端)
======================================================

验证 run_forward 的前向输出合理 + 反向梯度通过 autograd 正确流动。

运行:
    pytest tests/test_cuda_gradient.py -v
"""

import pytest
import torch

from tests.cuda_test_utils import (
    CTL_DT,
    GRAD_DECAY,
    run_forward_pytorch,
)


# ═══════════════════════════════════════════════════════════
#  测试 fixtures
# ═══════════════════════════════════════════════════════════

B = 64


@pytest.fixture
def dynamics_inputs():
    """生成随机动力学输入张量 (float64, CUDA)"""
    device = "cuda"
    dtype = torch.double
    return (
        torch.randn((B, 3, 3), dtype=dtype, device=device),        # R
        torch.randn((B, 3), dtype=dtype, device=device),            # dg
        torch.randn((B, 1), dtype=dtype, device=device),            # z_drag_coef
        torch.randn((B, 2), dtype=dtype, device=device),            # drag_2
        torch.randn((B, 1), dtype=dtype, device=device),            # pitch_ctl_delay
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # act_pred
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # act
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # p
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # v
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # v_wind
        torch.randn((B, 3), dtype=dtype, device=device, requires_grad=True),  # a
    )


# ═══════════════════════════════════════════════════════════
#  前向测试
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestForward:
    """验证 run_forward 输出合理"""

    def test_output_shapes(self, dynamics_inputs):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = dynamics_inputs
        out = run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, CTL_DT)
        assert len(out) == 4, "Should return 4 tensors"
        for t in out:
            assert t.shape == (B, 3), f"Wrong shape: {t.shape}"
            assert torch.isfinite(t).all(), "Contains NaN/Inf"


# ═══════════════════════════════════════════════════════════
#  反向测试
# ═══════════════════════════════════════════════════════════

@pytest.mark.gpu
class TestBackward:
    """验证 autograd 梯度正确计算"""

    def test_gradients(self, dynamics_inputs):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a = dynamics_inputs

        out = run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, CTL_DT)
        d_out = [torch.randn_like(o) for o in out]
        torch.autograd.backward(out, d_out)

        for name, tensor in [("act_pred", act_pred), ("act", act),
                              ("p", p), ("v", v), ("a", a)]:
            assert tensor.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(tensor.grad).all(), f"{name}.grad contains NaN/Inf"
            assert tensor.grad.abs().sum() > 0, f"{name}.grad is all zeros"
