"""
tests/conftest.py — pytest 配置、共享 fixtures、公共常量
=======================================================

提供:
  - TrainingConfig / TOPICS：与训练对齐的参数
  - pytest CLI 选项：--sim-timeout, --flight-duration, --device
  - session fixture：rclpy 生命周期、sim_data（预飞采集）
"""

import time
from dataclasses import dataclass

import pytest

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo, Imu
    from nav_msgs.msg import Odometry

    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False


# ═══════════════════════════════════════════════════════════
#  ROS2 话题定义
# ═══════════════════════════════════════════════════════════

TOPICS = {
    "depth":     "/uav1/rgbd_camera/depth_image",
    "caminfo":   "/uav1/rgbd_camera/camera_info",
    "odom":      "/uav1/odometry",
    "imu":       "/uav1/imu",
    "setpoint":  "/uav1/setpoint_raw/local",
    "collision": "/uav1/collision",
}


# ═══════════════════════════════════════════════════════════
#  训练配置期望值（来自 configs/single_agent.args）
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainingConfig:
    """与训练配置对齐的参数，部署/测试时使用"""

    # 相机
    fov_x_half_tan: float = 0.82
    cam_angle_deg: float = 20.0
    depth_width: int = 640
    depth_height: int = 480
    depth_encoding: str = "32FC1"
    depth_clip_far_min: float = 10.0

    # 频率下限 (Hz)
    min_hz_depth: float = 15.0
    min_hz_odom: float = 30.0
    min_hz_imu: float = 100.0

    # 坐标系
    odom_frame: str = "uav1/odom"
    body_frame: str = "uav1/base_link"

    # 控制
    control_hz: float = 15.0
    setpoint_qos: str = "BEST_EFFORT"

    # 模型
    dim_obs: int = 10
    dim_action: int = 6
    gru_hidden: int = 192

    # 部署默认值
    margin: float = 0.15
    max_speed: float = 2.0
    checkpoint: str = "checkpoints/checkpoint0004.pth"


TRAIN_CFG = TrainingConfig()


# ═══════════════════════════════════════════════════════════
#  数据采集基础设施
# ═══════════════════════════════════════════════════════════

@dataclass
class TopicStats:
    count: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    last_msg: object = None

    @property
    def elapsed(self) -> float:
        return self.last_time - self.first_time if self.count > 1 else 0.0

    @property
    def hz(self) -> float:
        return (self.count - 1) / self.elapsed if self.elapsed > 0 else 0.0


if HAS_RCLPY:
    class CollectorNode(Node):
        """订阅所有传感器话题，采集统计数据。不调用 rclpy.shutdown()。"""

        def __init__(self, timeout: float):
            super().__init__("preflight_checker")
            self.timeout = timeout
            self.stats: dict[str, TopicStats] = {}
            self.done = False

            qos_reliable = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )

            self._subs = [
                self.create_subscription(
                    Image, TOPICS["depth"], lambda m: self._cb("depth", m), qos_reliable),
                self.create_subscription(
                    CameraInfo, TOPICS["caminfo"], lambda m: self._cb("caminfo", m), qos_reliable),
                self.create_subscription(
                    Odometry, TOPICS["odom"], lambda m: self._cb("odom", m), qos_reliable),
                self.create_subscription(
                    Imu, TOPICS["imu"], lambda m: self._cb("imu", m), qos_reliable),
            ]

            self.start_time = time.monotonic()
            self.timer = self.create_timer(0.1, self._tick)

        def _cb(self, name: str, msg):
            now = time.monotonic()
            if name not in self.stats:
                self.stats[name] = TopicStats(first_time=now)
            s = self.stats[name]
            s.count += 1
            s.last_time = now
            s.last_msg = msg

        def _tick(self):
            if time.monotonic() - self.start_time > self.timeout:
                self.done = True


# ═══════════════════════════════════════════════════════════
#  pytest CLI 选项
# ═══════════════════════════════════════════════════════════

def pytest_addoption(parser):
    parser.addoption(
        "--sim-timeout", type=float, default=5.0,
        help="预飞检测数据采集时间（秒），默认 5",
    )
    parser.addoption(
        "--flight-duration", type=float, default=30.0,
        help="飞行测试持续时间（秒），默认 30",
    )
    parser.addoption(
        "--device", type=str, default="cuda",
        help="计算设备 (cuda / cpu)",
    )


# ═══════════════════════════════════════════════════════════
#  pytest fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def rclpy_context():
    """会话级 rclpy 生命周期管理"""
    if not HAS_RCLPY:
        pytest.skip("rclpy not available")
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="session")
def sim_data(rclpy_context, request):
    """
    预飞检测数据：采集 --sim-timeout 秒的传感器数据。

    返回 dict[str, TopicStats]，各 test_preflight 测试用例共享。
    """
    if not HAS_RCLPY:
        pytest.skip("rclpy not available")
    timeout = request.config.getoption("--sim-timeout")

    print(f"\n[fixture:sim_data] 正在采集仿真数据 ({timeout}s)...")
    node = CollectorNode(timeout=timeout)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    while not node.done:
        executor.spin_once(timeout_sec=0.1)

    data = dict(node.stats)
    node.destroy_node()
    executor.shutdown()

    summary = {k: v.count for k, v in data.items()}
    print(f"[fixture:sim_data] 采集完成: {summary}")
    return data


@pytest.fixture(scope="session")
def train_cfg():
    """训练配置（方便 fixture 注入）"""
    return TRAIN_CFG
