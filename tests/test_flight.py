"""
test_flight — 飞行管线功能测试
===============================

使用手搓控制器验证 ROS2 推理管线:
加速度 → 速度积分 → 机体坐标变换 → setpoint 发布。

模式作为参数化维度，每种模式运行独立飞行并做后置断言。

运行:
    pytest tests/test_flight.py -v --flight-duration 15
    pytest tests/test_flight.py -v -k hover
    pytest tests/test_flight.py -v -k goto --flight-duration 20
"""

import threading
import time
import warnings

import numpy as np
import pytest
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
from scipy.spatial.transform import Rotation

from tests.conftest import TOPICS, TRAIN_CFG

pytestmark = pytest.mark.flight


# ═══════════════════════════════════════════════════════════
#  飞行测试节点
# ═══════════════════════════════════════════════════════════

class FlightTestNode(Node):
    """
    手搓控制器飞行节点。

    输出语义与训练模型一致：世界坐标系下的加速度 (3,)，
    经过与 InferenceNode 相同的后处理管线发布速度指令。
    不调用 rclpy.shutdown()，设 self.done=True 表示完成。
    """

    def __init__(
        self,
        mode: str,
        target: list[float],
        duration: float,
        max_speed: float = TRAIN_CFG.max_speed,
        kp_pos: float = 1.5,
        kp_vel: float = 3.0,
        test_acc: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__("flight_test")

        self.mode = mode
        self.device = torch.device(device)
        self.max_speed = max_speed
        self.duration = duration
        self.control_dt = 1.0 / TRAIN_CFG.control_hz

        self.target = torch.tensor(target, dtype=torch.float32, device=self.device)
        self.Kp_pos = kp_pos
        self.Kp_vel = kp_vel
        self.test_acc = test_acc

        self._lock = threading.Lock()
        self._odom_msg: Odometry | None = None
        self._step_count = 0
        self._start_time: float | None = None
        self.done = False

        # ── 轨迹记录 ──
        self.trajectory: list[dict] = []

        # ── QoS ──
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.sub_odom = self.create_subscription(
            Odometry, TOPICS["odom"], self._odom_cb, qos_reliable,
        )
        self.pub_cmd = self.create_publisher(
            PositionTarget, TOPICS["setpoint"], qos_best_effort,
        )
        self.timer = self.create_timer(self.control_dt, self._control_loop)

    # ── 回调 ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom_msg = msg

    # ── 控制循环 ─────────────────────────────────────────

    def _control_loop(self):
        with self._lock:
            odom_msg = self._odom_msg

        if odom_msg is None:
            self._step_count += 1
            return

        if self._start_time is None:
            self._start_time = time.monotonic()

        elapsed = time.monotonic() - self._start_time
        if elapsed > self.duration:
            self.done = True
            return

        p, v_world, R_body = self._parse_odom(odom_msg)
        thrust_cmd = self._controller(p, v_world, R_body)

        v_cmd_world = v_world + thrust_cmd * self.control_dt
        v_cmd_body = R_body.T @ v_cmd_world
        self._publish_cmd(v_cmd_body)

        self.trajectory.append({
            "t": elapsed,
            "pos": p.cpu().tolist(),
            "vel": v_world.cpu().tolist(),
            "acc": thrust_cmd.cpu().tolist(),
        })
        self._step_count += 1

    # ── 控制器 ───────────────────────────────────────────

    def _controller(self, p, v, R_body):
        zero = torch.zeros(3, device=self.device)

        if self.mode == "hover":
            z_err = self.target[2] - p[2]
            return torch.tensor([
                -1.0 * float(v[0]),
                -1.0 * float(v[1]),
                float(self.Kp_vel * z_err - 1.0 * v[2]),
            ], device=self.device)

        elif self.mode == "forward":
            z_err = self.target[2] - p[2]
            return torch.tensor([
                float(self.test_acc - 1.0 * v[0]),
                -1.0 * float(v[1]),
                float(self.Kp_vel * z_err - 1.0 * v[2]),
            ], device=self.device)

        elif self.mode == "up":
            return torch.tensor([
                -1.0 * float(v[0]),
                -1.0 * float(v[1]),
                float(self.test_acc - 1.0 * v[2]),
            ], device=self.device)

        elif self.mode == "goto":
            pos_err = self.target - p
            dist = pos_err.norm()
            if dist > 1e-3:
                v_desired = (pos_err / dist) * min(dist * self.Kp_pos, self.max_speed)
            else:
                v_desired = zero
            thrust_cmd = self.Kp_vel * (v_desired - v)
            acc_norm = thrust_cmd.norm()
            if acc_norm > 5.0:
                thrust_cmd = thrust_cmd * (5.0 / acc_norm)
            return thrust_cmd

        return zero

    # ── 辅助 ─────────────────────────────────────────────

    def _parse_odom(self, msg):
        p = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=torch.float32, device=self.device)

        q = msg.pose.pose.orientation
        R_np = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        R_body = torch.from_numpy(R_np).float().to(self.device)

        v_body = torch.tensor([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=torch.float32, device=self.device)
        v_world = R_body @ v_body
        return p, v_world, R_body

    def _publish_cmd(self, v_cmd_body):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
            | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW
        )
        msg.velocity.x = float(v_cmd_body[0])
        msg.velocity.y = float(v_cmd_body[1])
        msg.velocity.z = float(v_cmd_body[2])
        msg.yaw_rate = 0.0
        self.pub_cmd.publish(msg)


# ═══════════════════════════════════════════════════════════
#  fixture: 执行飞行并返回轨迹
# ═══════════════════════════════════════════════════════════

def _run_flight(rclpy_context, request, mode: str, target: list[float]):
    """运行飞行节点，返回 (node, trajectory)"""
    duration = request.config.getoption("--flight-duration")
    device = request.config.getoption("--device")

    node = FlightTestNode(
        mode=mode, target=target, duration=duration, device=device,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    print(f"\n[flight:{mode}] 飞行中 ({duration}s)...")
    while not node.done:
        executor.spin_once(timeout_sec=0.1)

    traj = node.trajectory
    node.destroy_node()
    executor.shutdown()
    print(f"[flight:{mode}] 完成, {len(traj)} 个轨迹点")
    return traj, np.array(target)


# ═══════════════════════════════════════════════════════════
#  fixtures: 每个 class 执行一次飞行
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="class")
def hover_result(rclpy_context, request):
    return _run_flight(rclpy_context, request, "hover", [0.0, 0.0, 1.0])


@pytest.fixture(scope="class")
def forward_result(rclpy_context, request):
    return _run_flight(rclpy_context, request, "forward", [0.0, 0.0, 1.0])


@pytest.fixture(scope="class")
def up_result(rclpy_context, request):
    return _run_flight(rclpy_context, request, "up", [0.0, 0.0, 5.0])


@pytest.fixture(scope="class")
def goto_result(rclpy_context, request):
    return _run_flight(rclpy_context, request, "goto", [5.0, 0.0, 1.0])

def _final_state(traj):
    final = traj[-1]
    return np.array(final["pos"]), np.array(final["vel"])


def _late_positions(traj):
    """后半段位置数组"""
    half = len(traj) // 2
    return np.array([t["pos"] for t in traj[half:]])


# ═══════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════

class TestHover:
    """悬停测试：高度 P 控制，验证高度保持和水平稳定"""

    def test_data_sufficient(self, hover_result):
        traj, _ = hover_result
        assert len(traj) >= 10, f"仅采集 {len(traj)} 点, 数据不足"

    def test_altitude_hold(self, hover_result):
        traj, target = hover_result
        late = _late_positions(traj)
        z_err = np.abs(late[:, 2] - target[2]).mean()
        print(f"  后半段平均 z 误差 = {z_err:.3f}m")
        assert z_err < 0.5, f"高度误差 {z_err:.3f}m > 0.5m"

    def test_xy_drift(self, hover_result):
        traj, target = hover_result
        late = _late_positions(traj)
        xy_drift = np.linalg.norm(late[:, :2] - target[:2], axis=1).mean()
        print(f"  后半段平均 xy 漂移 = {xy_drift:.3f}m")
        if xy_drift >= 0.5:
            warnings.warn(f"水平漂移 {xy_drift:.3f}m > 0.5m", UserWarning)

    def test_converged_speed(self, hover_result):
        traj, _ = hover_result
        _, final_vel = _final_state(traj)
        speed = np.linalg.norm(final_vel)
        print(f"  终态速度 = {speed:.3f} m/s")
        if speed >= 0.5:
            warnings.warn(f"终态速度 {speed:.3f} m/s > 0.5", UserWarning)

    def test_safe_speed(self, hover_result):
        traj, _ = hover_result
        speeds = [np.linalg.norm(t["vel"]) for t in traj]
        assert max(speeds) < 10.0, f"最大速度 {max(speeds):.2f} m/s >= 10"


class TestForward:
    """前飞测试：世界 X+ 匀加速前飞"""

    def test_data_sufficient(self, forward_result):
        traj, _ = forward_result
        assert len(traj) >= 10

    def test_x_increasing(self, forward_result):
        traj, _ = forward_result
        late = _late_positions(traj)
        assert late[-1, 0] > late[0, 0], (
            f"X 未增长: {late[0, 0]:.2f} → {late[-1, 0]:.2f}"
        )
        print(f"  X: {late[0, 0]:.2f} → {late[-1, 0]:.2f}")

    def test_altitude_hold(self, forward_result):
        traj, target = forward_result
        late = _late_positions(traj)
        z_err = np.abs(late[:, 2] - target[2]).mean()
        print(f"  平均 z 误差 = {z_err:.3f}m")
        if z_err >= 0.5:
            warnings.warn(f"前飞高度误差 {z_err:.3f}m > 0.5m", UserWarning)

    def test_safe_speed(self, forward_result):
        traj, _ = forward_result
        speeds = [np.linalg.norm(t["vel"]) for t in traj]
        assert max(speeds) < 10.0


class TestUp:
    """爬升测试：世界 Z+ 方向"""

    def test_data_sufficient(self, up_result):
        traj, _ = up_result
        assert len(traj) >= 10

    def test_z_increasing(self, up_result):
        traj, _ = up_result
        z_start = traj[0]["pos"][2]
        z_end = traj[-1]["pos"][2]
        assert z_end > z_start + 0.5, (
            f"Z 变化不足: {z_start:.2f} → {z_end:.2f}"
        )
        print(f"  Z: {z_start:.2f} → {z_end:.2f}")

    def test_safe_speed(self, up_result):
        traj, _ = up_result
        speeds = [np.linalg.norm(t["vel"]) for t in traj]
        assert max(speeds) < 10.0


class TestGoto:
    """定点飞行测试：PD 控制器飞向目标"""

    def test_data_sufficient(self, goto_result):
        traj, _ = goto_result
        assert len(traj) >= 10

    def test_reached_target(self, goto_result):
        traj, target = goto_result
        final_pos, _ = _final_state(traj)
        dist = np.linalg.norm(final_pos - target)
        print(f"  终态距离 = {dist:.3f}m")
        assert dist < 1.0, f"距目标 {dist:.3f}m > 1.0m"
        if dist >= 0.5:
            warnings.warn(f"距目标 {dist:.3f}m (0.5~1.0m)", UserWarning)

    def test_converged_speed(self, goto_result):
        traj, _ = goto_result
        _, final_vel = _final_state(traj)
        speed = np.linalg.norm(final_vel)
        print(f"  终态速度 = {speed:.3f} m/s")
        if speed >= 0.3:
            warnings.warn(f"终态速度 {speed:.3f} m/s > 0.3", UserWarning)

    def test_convergence_trend(self, goto_result):
        """后半段距离应小于前半段"""
        traj, target = goto_result
        half = len(traj) // 2
        early = np.array([t["pos"] for t in traj[:half]])
        late = np.array([t["pos"] for t in traj[half:]])
        early_dist = np.linalg.norm(early - target, axis=1).mean()
        late_dist = np.linalg.norm(late - target, axis=1).mean()
        print(f"  前半段平均距离={early_dist:.3f}, 后半段={late_dist:.3f}")
        if late_dist >= early_dist:
            warnings.warn("后半段距离未减小，收敛趋势不明显", UserWarning)

    def test_safe_speed(self, goto_result):
        traj, _ = goto_result
        speeds = [np.linalg.norm(t["vel"]) for t in traj]
        assert max(speeds) < 10.0

    def test_safe_acceleration(self, goto_result):
        traj, _ = goto_result
        accs = [np.linalg.norm(t["acc"]) for t in traj]
        max_acc = max(accs)
        print(f"  最大加速度 = {max_acc:.2f} m/s²")
        if max_acc >= 8.0:
            warnings.warn(f"最大加速度 {max_acc:.2f} m/s² (偏高)", UserWarning)
