"""
test_preflight — 仿真环境预飞检测
==================================

每项检查 = 一个独立的 pytest test，共享 sim_data session fixture。

运行:
    pytest tests/test_preflight.py -v
    pytest tests/test_preflight.py -v --sim-timeout 3
    pytest tests/test_preflight.py -k "depth" -v      # 只跑深度图相关
"""

import math
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.conftest import TOPICS, TRAIN_CFG


# ═══════════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════════

pytestmark = pytest.mark.preflight


def _require_topic(sim_data, key: str):
    """断言话题存在且有消息，返回 stats 对象"""
    assert key in sim_data and sim_data[key].count > 0, (
        f"话题 {TOPICS[key]} 未收到任何消息"
    )
    return sim_data[key]


# ═══════════════════════════════════════════════════════════
#  1. 话题存在性
# ═══════════════════════════════════════════════════════════

class TestTopicAvailability:
    """验证所有必需话题已在仿真中发布"""

    @pytest.mark.parametrize("key", ["depth", "caminfo", "odom", "imu"])
    def test_topic_active(self, sim_data, key):
        stats = sim_data.get(key)
        assert stats is not None and stats.count > 0, (
            f"话题 {TOPICS[key]} 未收到任何消息"
        )
        # 顺便打印采集量
        print(f"  {TOPICS[key]}: {stats.count} msgs")


# ═══════════════════════════════════════════════════════════
#  2. 频率检测
# ═══════════════════════════════════════════════════════════

class TestFrequency:
    """验证传感器发布频率满足最低要求"""

    @pytest.mark.parametrize("key,min_hz,label", [
        ("depth", TRAIN_CFG.min_hz_depth, "深度图"),
        ("odom",  TRAIN_CFG.min_hz_odom,  "里程计"),
        ("imu",   TRAIN_CFG.min_hz_imu,   "IMU"),
    ])
    def test_frequency(self, sim_data, key, min_hz, label):
        stats = _require_topic(sim_data, key)
        hz = stats.hz
        assert hz >= min_hz, (
            f"{label} 频率 {hz:.1f} Hz < 下限 {min_hz:.0f} Hz"
        )
        print(f"  {label}: {hz:.1f} Hz (>= {min_hz:.0f})")


# ═══════════════════════════════════════════════════════════
#  3. 深度图格式
# ═══════════════════════════════════════════════════════════

class TestDepthImage:
    """验证深度图分辨率、编码、数据完整性"""

    def _get_depth_msg(self, sim_data):
        stats = _require_topic(sim_data, "depth")
        return stats.last_msg

    def test_resolution(self, sim_data):
        msg = self._get_depth_msg(sim_data)
        cfg = TRAIN_CFG
        assert msg.width == cfg.depth_width and msg.height == cfg.depth_height, (
            f"分辨率 {msg.width}x{msg.height}, 期望 {cfg.depth_width}x{cfg.depth_height}"
        )

    def test_encoding(self, sim_data):
        msg = self._get_depth_msg(sim_data)
        assert msg.encoding == TRAIN_CFG.depth_encoding, (
            f"编码 {msg.encoding}, 期望 {TRAIN_CFG.depth_encoding}"
        )

    def test_data_size(self, sim_data):
        msg = self._get_depth_msg(sim_data)
        expected = msg.width * msg.height * 4  # float32
        assert len(msg.data) == expected, (
            f"数据 {len(msg.data)} bytes, 期望 {expected}"
        )

    def test_depth_values_valid(self, sim_data):
        msg = self._get_depth_msg(sim_data)
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        valid = depth[np.isfinite(depth)]
        assert len(valid) > 0, "深度图全部为 inf/nan，传感器可能未工作"
        d_min, d_max = valid.min(), valid.max()
        n_inf = np.isinf(depth).sum()
        n_nan = np.isnan(depth).sum()
        print(f"  有效深度范围: [{d_min:.3f}, {d_max:.3f}] m, inf={n_inf}, nan={n_nan}")

    def test_clip_far(self, sim_data):
        msg = self._get_depth_msg(sim_data)
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        valid = depth[np.isfinite(depth)]
        if len(valid) == 0:
            pytest.skip("无有效深度数据")
        d_max = valid.max()
        if d_max < TRAIN_CFG.depth_clip_far_min:
            warnings.warn(
                f"最大有效深度 {d_max:.1f}m < {TRAIN_CFG.depth_clip_far_min}m，"
                "clip_far 可能过小",
                UserWarning,
            )


# ═══════════════════════════════════════════════════════════
#  4. 相机内参 & FOV
# ═══════════════════════════════════════════════════════════

class TestCamera:
    """验证相机内参、FOV、宽高比"""

    def _get_caminfo(self, sim_data):
        stats = _require_topic(sim_data, "caminfo")
        return stats.last_msg

    def test_intrinsics_nonzero(self, sim_data):
        msg = self._get_caminfo(sim_data)
        fx, fy = msg.k[0], msg.k[4]
        assert fx > 0 and fy > 0, f"fx={fx}, fy={fy}，内参无效"
        cx, cy = msg.k[2], msg.k[5]
        print(f"  内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")

    def test_horizontal_fov(self, sim_data):
        msg = self._get_caminfo(sim_data)
        cfg = TRAIN_CFG
        fx = msg.k[0]
        fov_x_half_tan = (msg.width / 2) / fx
        fov_err = abs(fov_x_half_tan - cfg.fov_x_half_tan) / cfg.fov_x_half_tan

        hfov_deg = math.degrees(2 * math.atan(fov_x_half_tan))
        expected_hfov_deg = math.degrees(2 * math.atan(cfg.fov_x_half_tan))

        assert fov_err < 0.15, (
            f"fov_x_half_tan={fov_x_half_tan:.4f} (HFOV={hfov_deg:.1f}°), "
            f"期望 {cfg.fov_x_half_tan} ({expected_hfov_deg:.1f}°), "
            f"误差 {fov_err*100:.1f}% > 15%"
        )
        if fov_err >= 0.05:
            warnings.warn(
                f"FOV 误差 {fov_err*100:.1f}% (5~15%), "
                f"实际 {fov_x_half_tan:.4f} vs 期望 {cfg.fov_x_half_tan}",
                UserWarning,
            )
        print(f"  HFOV={hfov_deg:.1f}° (期望 {expected_hfov_deg:.1f}°), 误差 {fov_err*100:.1f}%")

    def test_aspect_ratio(self, sim_data):
        msg = self._get_caminfo(sim_data)
        cfg = TRAIN_CFG
        aspect = msg.width / msg.height
        expected = cfg.depth_width / cfg.depth_height
        assert abs(aspect - expected) < 0.01, (
            f"宽高比 {aspect:.4f} != 期望 {expected:.4f}"
        )

    def test_frame_id(self, sim_data):
        msg = self._get_caminfo(sim_data)
        print(f"  frame_id: {msg.header.frame_id}")
        # 仅记录，不强制断言（光学帧名可能变化）

    def test_pitch_angle_reminder(self, sim_data):
        """相机安装俯仰角无法从话题自动验证"""
        cfg = TRAIN_CFG
        warnings.warn(
            f"期望上仰 {cfg.cam_angle_deg:.0f}°, 无法从话题自动验证, "
            f"请在 Jinja 模板或 SDF 中确认 pitch = -{cfg.cam_angle_deg:.0f}°",
            UserWarning,
        )


# ═══════════════════════════════════════════════════════════
#  5. 里程计
# ═══════════════════════════════════════════════════════════

class TestOdometry:
    """验证里程计帧 ID、姿态、坐标系"""

    def _get_odom(self, sim_data):
        stats = _require_topic(sim_data, "odom")
        return stats.last_msg

    def test_frame_id(self, sim_data):
        msg = self._get_odom(sim_data)
        cfg = TRAIN_CFG
        assert cfg.odom_frame in msg.header.frame_id, (
            f"frame_id '{msg.header.frame_id}' 不含 '{cfg.odom_frame}'"
        )

    def test_child_frame_id(self, sim_data):
        msg = self._get_odom(sim_data)
        cfg = TRAIN_CFG
        assert cfg.body_frame in msg.child_frame_id, (
            f"child_frame_id '{msg.child_frame_id}' 不含 '{cfg.body_frame}'"
        )

    def test_quaternion_unit(self, sim_data):
        msg = self._get_odom(sim_data)
        q = msg.pose.pose.orientation
        qnorm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert abs(qnorm - 1.0) < 0.01, f"姿态四元数范数 {qnorm:.6f}，非单位四元数"

    def test_flu_coordinate_system(self, sim_data):
        """机体 Z 轴应朝上（FLU 坐标系）"""
        msg = self._get_odom(sim_data)
        q = msg.pose.pose.orientation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        body_up = R[:, 2]

        print(f"  机体 Z 轴(世界系): [{body_up[0]:.4f}, {body_up[1]:.4f}, {body_up[2]:.4f}]")

        assert body_up[2] > -0.9, (
            f"机体 Z 轴朝下 (z={body_up[2]:.3f}), 可能是 NED/FRD 坐标系!"
        )
        if body_up[2] <= 0.9:
            warnings.warn(
                f"机体 Z 轴偏离竖直 (z={body_up[2]:.3f}), 无人机可能不在悬停",
                UserWarning,
            )

    def test_state_report(self, sim_data):
        """打印当前位置和速度（信息项，总是通过）"""
        msg = self._get_odom(sim_data)
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        print(f"  位置: [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] m")
        print(f"  速度(body): [{v.x:.3f}, {v.y:.3f}, {v.z:.3f}] m/s, |v|={speed:.3f}")


# ═══════════════════════════════════════════════════════════
#  6. IMU
# ═══════════════════════════════════════════════════════════

class TestIMU:
    """验证 IMU 姿态、重力方向"""

    def _get_imu(self, sim_data):
        stats = _require_topic(sim_data, "imu")
        return stats.last_msg

    def test_quaternion_unit(self, sim_data):
        msg = self._get_imu(sim_data)
        q = msg.orientation
        qnorm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert abs(qnorm - 1.0) < 0.01, f"IMU 姿态四元数范数 {qnorm:.6f}，非单位四元数"

    def test_angular_velocity(self, sim_data):
        msg = self._get_imu(sim_data)
        av = msg.angular_velocity
        print(f"  角速度: [{av.x:.4f}, {av.y:.4f}, {av.z:.4f}] rad/s")

    def test_gravity_direction(self, sim_data):
        """FLU 坐标系下 acc_z 应 > 0（重力反向 = 向上）"""
        msg = self._get_imu(sim_data)
        la = msg.linear_acceleration
        acc_mag = math.sqrt(la.x**2 + la.y**2 + la.z**2)
        print(f"  线性加速度: [{la.x:.3f}, {la.y:.3f}, {la.z:.3f}] m/s², |a|={acc_mag:.3f}")

        assert la.z > -8.0, (
            f"acc_z={la.z:.2f} < -8.0, 可能是 FRD/NED 坐标系!"
        )
        if la.z <= 8.0:
            warnings.warn(
                f"acc_z={la.z:.2f}, 无法明确判断重力方向, 无人机可能不在悬停",
                UserWarning,
            )


# ═══════════════════════════════════════════════════════════
#  7. 控制接口
# ═══════════════════════════════════════════════════════════

class TestControlInterface:
    """验证 setpoint 话题配置"""

    def test_setpoint_qos_reminder(self, sim_data):
        cfg = TRAIN_CFG
        warnings.warn(
            f"setpoint 话题 {TOPICS['setpoint']} QoS 需设为 {cfg.setpoint_qos}, "
            "请确认飞控端已订阅",
            UserWarning,
        )
