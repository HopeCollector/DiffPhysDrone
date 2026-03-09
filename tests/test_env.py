"""
test_env — 仿真环境单元测试
============================

验证 Env 类的基本功能：构造、reset、render、run、find_vec_to_nearest_pt。

需要 CUDA GPU。

运行:
    pytest tests/test_env.py -v
"""

import pytest
import torch

from bitpilot.env import Env


B = 8
W, H = 64, 48
GRAD_DECAY = 0.4
CTL_DT = 1 / 15


@pytest.fixture(scope="module")
def env():
    """构造一个小批量环境实例"""
    device = "cuda" if torch.cuda.is_available() else pytest.skip("需要 CUDA GPU")
    return Env(B, W, H, GRAD_DECAY, device, single=True, speed_mtp=4)


@pytest.mark.gpu
class TestEnvBasic:
    """Env 基本生命周期测试"""

    def test_construction(self, env):
        assert env.batch_size == B

    def test_reset_state_shapes(self, env):
        env.reset()
        assert env.p.shape == (B, 3)
        assert env.v.shape == (B, 3)
        assert env.a.shape == (B, 3)
        assert env.R.shape == (B, 3, 3)
        assert env.p_target.shape == (B, 3)

    def test_reset_position_finite(self, env):
        env.reset()
        assert torch.isfinite(env.p).all()
        assert torch.isfinite(env.v).all()


@pytest.mark.gpu
class TestEnvRender:
    """Env.render 输出验证"""

    def test_render_shape(self, env):
        env.reset()
        depth, flow = env.render(CTL_DT)
        assert depth.shape == (B, H, W)

    def test_render_values_positive(self, env):
        env.reset()
        depth, _ = env.render(CTL_DT)
        assert (depth >= 0).all()

    def test_render_no_nan(self, env):
        env.reset()
        depth, _ = env.render(CTL_DT)
        assert torch.isfinite(depth).all()


@pytest.mark.gpu
class TestEnvRun:
    """Env.run 物理推进验证"""

    def test_run_updates_position(self, env):
        env.reset()
        p_before = env.p.clone()
        act = torch.randn(B, 3, device=env.device) * 0.1
        v_pred = torch.randn(B, 3, device=env.device)
        env.run(act, CTL_DT, v_pred)
        # 位置应该有变化（除非速度和加速度恰好为零，概率极低）
        assert not torch.equal(env.p, p_before)

    def test_run_state_finite(self, env):
        env.reset()
        act = torch.randn(B, 3, device=env.device) * 0.1
        v_pred = torch.randn(B, 3, device=env.device)
        env.run(act, CTL_DT, v_pred)
        assert torch.isfinite(env.p).all()
        assert torch.isfinite(env.v).all()
        assert torch.isfinite(env.a).all()


@pytest.mark.gpu
class TestEnvCollision:
    """Env.find_vec_to_nearest_pt 验证"""

    def test_vec_to_nearest_pt_shape(self, env):
        env.reset()
        vec = env.find_vec_to_nearest_pt()
        # 返回 (sub_steps, B, 3)
        assert vec.ndim == 3
        assert vec.shape[1] == B
        assert vec.shape[2] == 3

    def test_vec_to_nearest_pt_finite(self, env):
        env.reset()
        vec = env.find_vec_to_nearest_pt()
        assert torch.isfinite(vec).all()
