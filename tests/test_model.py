"""
test_model — 模型单元测试
=========================

验证 Model 的前向传播形状、hidden state 传递、checkpoint 加载。

运行:
    pytest tests/test_model.py -v
"""

import pytest
import torch

from bitpilot.model import Model


# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

DIM_OBS = 10
DIM_ACTION = 6
GRU_HIDDEN = 192


# ═══════════════════════════════════════════════════════════
#  前向传播形状
# ═══════════════════════════════════════════════════════════

class TestForwardShape:
    """验证模型输入输出张量形状"""

    @pytest.fixture
    def model(self):
        return Model(dim_obs=DIM_OBS, dim_action=DIM_ACTION).eval()

    def test_single_sample(self, model):
        x = torch.randn(1, 1, 12, 16)
        state = torch.randn(1, DIM_OBS)
        act, _, hx = model(x, state, None)
        assert act.shape == (1, DIM_ACTION)
        assert hx.shape == (1, GRU_HIDDEN)

    def test_batch(self, model):
        B = 64
        x = torch.randn(B, 1, 12, 16)
        state = torch.randn(B, DIM_OBS)
        act, _, hx = model(x, state, None)
        assert act.shape == (B, DIM_ACTION)
        assert hx.shape == (B, GRU_HIDDEN)

    def test_action_range_reasonable(self, model):
        """模型输出不应该是 NaN 或 Inf"""
        x = torch.randn(4, 1, 12, 16)
        state = torch.randn(4, DIM_OBS)
        act, _, _ = model(x, state, None)
        assert torch.isfinite(act).all()


# ═══════════════════════════════════════════════════════════
#  Hidden State 传递
# ═══════════════════════════════════════════════════════════

class TestHiddenState:
    """验证 GRU hidden state 的正确传递"""

    @pytest.fixture
    def model(self):
        return Model(dim_obs=DIM_OBS, dim_action=DIM_ACTION).eval()

    def test_hx_passthrough(self, model):
        """传入 hx 后输出的 hx 应与不传 hx 不同"""
        x = torch.randn(1, 1, 12, 16)
        state = torch.randn(1, DIM_OBS)

        _, _, hx_none = model(x, state, None)
        _, _, hx_with = model(x, state, hx_none)
        assert not torch.equal(hx_none, hx_with)

    def test_sequential_consistency(self, model):
        """同样的输入序列应产生确定性输出"""
        x = torch.randn(1, 1, 12, 16)
        state = torch.randn(1, DIM_OBS)

        # Run 1
        hx = None
        acts_1 = []
        for _ in range(5):
            act, _, hx = model(x, state, hx)
            acts_1.append(act.clone())

        # Run 2
        hx = None
        acts_2 = []
        for _ in range(5):
            act, _, hx = model(x, state, hx)
            acts_2.append(act.clone())

        for a1, a2 in zip(acts_1, acts_2):
            assert torch.equal(a1, a2)


# ═══════════════════════════════════════════════════════════
#  Checkpoint 加载
# ═══════════════════════════════════════════════════════════

class TestCheckpoint:
    """验证 checkpoint 文件加载"""

    CKPT_PATH = "checkpoints/checkpoint0004.pth"

    def test_load_checkpoint(self):
        model = Model(dim_obs=DIM_OBS, dim_action=DIM_ACTION)
        state_dict = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    def test_inference_after_load(self):
        model = Model(dim_obs=DIM_OBS, dim_action=DIM_ACTION)
        state_dict = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        x = torch.randn(1, 1, 12, 16)
        state = torch.randn(1, DIM_OBS)
        act, _, hx = model(x, state, None)
        assert torch.isfinite(act).all()
        assert torch.isfinite(hx).all()
