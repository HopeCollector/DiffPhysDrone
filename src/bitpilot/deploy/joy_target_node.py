#!/usr/bin/env python3
"""
手柄 → 目标位置 转换节点

将手柄摇杆量映射为当前位置的偏移，生成期望目标位姿发布到 /uav1/target_pose。

摇杆映射（标准手柄布局）:
  右摇杆 前后 (axes[4]) → X 偏移（前+），最大 ±5 m
  右摇杆 左右 (axes[3]) → Y 偏移（左+），最大 ±5 m
  左摇杆 前后 (axes[1]) → Z 偏移（上+），最大 ±2 m
  左摇杆 左右 (axes[0]) → 航向偏移（左+），最大 ±90°

偏移在无人机当前航向坐标系下计算，叠加到当前里程计位置后发布。

订阅:
  /joy             (sensor_msgs/Joy)      — 手柄消息
  /uav1/odometry   (nav_msgs/Odometry)    — 里程计

发布:
  /uav1/target_pose (geometry_msgs/PoseStamped) — 目标位姿

用法:
  source /opt/ros/jazzy/setup.bash
  uv run bitpilot-joy
"""

import argparse
import math
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation


class JoyTargetNode(Node):
    """将手柄输入映射为目标位姿并发布"""

    # 标准手柄轴索引
    AXIS_LEFT_X = 0   # 左摇杆 水平（左=+1）
    AXIS_LEFT_Y = 1   # 左摇杆 垂直（上=+1）
    AXIS_RIGHT_X = 3  # 右摇杆 水平（左=+1）
    AXIS_RIGHT_Y = 4  # 右摇杆 垂直（上=+1）

    def __init__(self, args):
        super().__init__("joy_target")

        # ── 参数 ──
        self.max_xy = args.max_xy          # m，水平偏移上限
        self.max_z = args.max_z            # m，垂直偏移上限
        self.max_yaw = math.radians(args.max_yaw_deg)  # rad，航向偏移上限
        self.deadzone = args.deadzone      # 摇杆死区
        self.publish_rate = args.rate       # Hz

        # ── 数据缓存 ──
        self._lock = threading.Lock()
        self._joy_msg: Joy | None = None
        self._odom_msg: Odometry | None = None

        # ── QoS ──
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 订阅 ──
        self.sub_joy = self.create_subscription(
            Joy, "/joy", self._joy_cb, qos_reliable,
        )
        self.sub_odom = self.create_subscription(
            Odometry, "/uav1/odometry", self._odom_cb, qos_reliable,
        )

        # ── 发布 ──
        self.pub_target = self.create_publisher(
            PoseStamped, "/uav1/target_pose", qos_reliable,
        )

        # ── 定时器 ──
        self.timer = self.create_timer(1.0 / self.publish_rate, self._publish_loop)
        self._step = 0

        self.get_logger().info(
            f"手柄目标节点启动 | max_xy={self.max_xy}m max_z={self.max_z}m "
            f"max_yaw={args.max_yaw_deg}° deadzone={self.deadzone} rate={self.publish_rate}Hz"
        )

    # ── 回调 ──

    def _joy_cb(self, msg: Joy):
        with self._lock:
            self._joy_msg = msg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom_msg = msg

    # ── 工具方法 ──

    def _apply_deadzone(self, value: float) -> float:
        """应用死区：小于阈值归零，超出部分线性映射到 [0, 1]"""
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    # ── 主发布循环 ──

    def _publish_loop(self):
        with self._lock:
            joy_msg = self._joy_msg
            odom_msg = self._odom_msg

        if odom_msg is None:
            if self._step % (self.publish_rate * 3) == 0:
                self.get_logger().warn("等待里程计数据...")
            self._step += 1
            return

        if joy_msg is None:
            if self._step % (self.publish_rate * 3) == 0:
                self.get_logger().warn("等待手柄数据...")
            self._step += 1
            return

        # ── 解析摇杆量（带死区） ──
        axes = joy_msg.axes
        if len(axes) < 5:
            self.get_logger().error(f"手柄轴数不足: {len(axes)} < 5", throttle_duration_sec=5.0)
            return

        raw_right_y = self._apply_deadzone(axes[self.AXIS_RIGHT_Y])  # 前后 → X
        raw_right_x = self._apply_deadzone(axes[self.AXIS_RIGHT_X])  # 左右 → Y
        raw_left_y = self._apply_deadzone(axes[self.AXIS_LEFT_Y])    # 上下 → Z
        raw_left_x = self._apply_deadzone(axes[self.AXIS_LEFT_X])    # 左右 → Yaw

        # 映射到偏移量
        dx_heading = raw_right_y * self.max_xy   # 航向坐标系 X（前）
        dy_heading = raw_right_x * self.max_xy   # 航向坐标系 Y（左）
        dz_world = raw_left_y * self.max_z       # 世界坐标系 Z（上）
        dyaw = raw_left_x * self.max_yaw         # 航向偏移（左+，rad）

        # ── 解析里程计 ──
        pos = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        p_current = np.array([pos.x, pos.y, pos.z])

        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        R_body = rot.as_matrix()  # (3, 3)

        # 当前航向（从机体 X 轴水平投影提取 yaw）
        fwd = R_body[:, 0].copy()
        fwd[2] = 0.0
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-6:
            fwd /= fwd_norm
        else:
            fwd = np.array([1.0, 0.0, 0.0])
        current_yaw = math.atan2(fwd[1], fwd[0])

        # ── 航向坐标系下偏移 → 世界坐标系 ──
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        dx_world = dx_heading * cos_yaw - dy_heading * sin_yaw
        dy_world = dx_heading * sin_yaw + dy_heading * cos_yaw

        # ── 目标位置 = 当前位置 + 偏移 ──
        target_x = p_current[0] + dx_world
        target_y = p_current[1] + dy_world
        target_z = p_current[2] + dz_world

        # ── 目标航向 = 当前航向 + 偏移 ──
        target_yaw = current_yaw + dyaw

        # ── 构建并发布 PoseStamped ──
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(target_x)
        msg.pose.position.y = float(target_y)
        msg.pose.position.z = float(target_z)

        # yaw → 四元数（仅 yaw，pitch=roll=0）
        target_q = Rotation.from_euler('z', target_yaw).as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = float(target_q[0])
        msg.pose.orientation.y = float(target_q[1])
        msg.pose.orientation.z = float(target_q[2])
        msg.pose.orientation.w = float(target_q[3])

        self.pub_target.publish(msg)

        # ── 低频日志 ──
        self._step += 1
        if self._step % (self.publish_rate * 2) == 0:
            dist = math.sqrt(dx_world**2 + dy_world**2 + dz_world**2)
            self.get_logger().info(
                f"target=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) "
                f"yaw={math.degrees(target_yaw):.1f}° "
                f"offset={dist:.2f}m "
                f"stick=({raw_right_y:+.2f},{raw_right_x:+.2f},{raw_left_y:+.2f},{raw_left_x:+.2f})"
            )


def main():
    parser = argparse.ArgumentParser(description="手柄 → 目标位置 转换节点")
    parser.add_argument(
        "--max_xy", type=float, default=5.0,
        help="水平偏移最大值 (m)，对应摇杆满量程",
    )
    parser.add_argument(
        "--max_z", type=float, default=2.0,
        help="垂直偏移最大值 (m)，对应摇杆满量程",
    )
    parser.add_argument(
        "--max_yaw_deg", type=float, default=90.0,
        help="航向偏移最大值 (°)，对应摇杆满量程",
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.08,
        help="摇杆死区阈值 (0~1)",
    )
    parser.add_argument(
        "--rate", type=float, default=15.0,
        help="发布频率 (Hz)",
    )

    args, _ = parser.parse_known_args()

    rclpy.init()
    node = JoyTargetNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中断信号，节点退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
