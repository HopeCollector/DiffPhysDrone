#!/usr/bin/env python3
"""
DiffPhy 模型 ROS2 推理节点

将训练好的视觉敏捷飞行模型部署到 Gazebo 仿真环境中。

订阅:
  /uav1/rgbd_camera/depth_image  (sensor_msgs/Image)   — 深度图
  /uav1/odometry                 (nav_msgs/Odometry)    — 里程计

发布:
  /uav1/setpoint_raw/local       (mavros_msgs/PositionTarget) — 速度指令

用法:
  source /opt/ros/jazzy/setup.bash
  python3 inference_node.py [--checkpoint PATH] [--target X Y Z] [--max_speed V] ...

飞行管线测试请使用:
  python -m tests.run flight hover
  python -m tests.run flight goto --target 5 0 1
"""

import argparse
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget

from scipy.spatial.transform import Rotation

# ────────────────────────── 导入模型 ──────────────────────────
sys.path.insert(0, "/ws")
from model import Model


class InferenceNode(Node):
    """基于训练模型的 15 Hz 推理节点"""

    def __init__(self, args):
        super().__init__("diffphy_inference")

        # ── 参数 ──────────────────────────────────────────────
        self.device = torch.device(args.device)
        self.max_speed = args.max_speed
        self.margin = args.margin
        self.control_dt = 1.0 / 15.0  # 控制周期 ~15 Hz
        self.target = torch.tensor(args.target, dtype=torch.float32, device=self.device)
        self.gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float32, device=self.device)

        # ── 加载模型 ──
        self.model = Model(dim_obs=10, dim_action=6).to(self.device)
        state_dict = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.get_logger().info(f"模型加载完成: {args.checkpoint}")

        # ── GRU 隐状态 ───────────────────────────────────────
        self.hx: torch.Tensor | None = None

        # ── 数据缓存（加锁保护） ─────────────────────────────
        self._lock = threading.Lock()
        self._depth_msg: Image | None = None
        self._odom_msg: Odometry | None = None

        # ── QoS 配置 ─────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 订阅 ─────────────────────────────────────────────
        self.sub_depth = self.create_subscription(
            Image, "/uav1/rgbd_camera/depth_image",
            self._depth_cb, qos_reliable,
        )
        self.sub_odom = self.create_subscription(
            Odometry, "/uav1/odometry",
            self._odom_cb, qos_reliable,
        )

        # ── 发布 ─────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(
            PositionTarget, "/uav1/setpoint_raw/local", qos_best_effort,
        )

        # ── 15 Hz 定时器 ─────────────────────────────────────
        self.timer = self.create_timer(self.control_dt, self._control_loop)
        self._step_count = 0

        self.get_logger().info(
            f"推理节点启动 | 目标: {args.target} | max_speed: {args.max_speed} m/s | margin: {args.margin} m"
        )

    # ── 回调：仅缓存最新数据 ─────────────────────────────────

    def _depth_cb(self, msg: Image):
        with self._lock:
            self._depth_msg = msg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom_msg = msg

    # ── 主控制循环 ────────────────────────────────────────────

    @torch.no_grad()
    def _control_loop(self):
        # ── 1. 取出缓存数据 ──
        with self._lock:
            depth_msg = self._depth_msg
            odom_msg = self._odom_msg

        if odom_msg is None:
            if self._step_count % 30 == 0:
                self.get_logger().warn("等待里程计数据...")
            self._step_count += 1
            return

        if depth_msg is None:
            if self._step_count % 30 == 0:
                self.get_logger().warn("等待深度图数据...")
            self._step_count += 1
            return

        # ── 2. 解析里程计 ──
        p_current, v_world, R_body = self._parse_odom(odom_msg)

        # ── 3. 模型推理 ──
        depth_raw = self._parse_depth(depth_msg)
        x = self._preprocess_depth(depth_raw)            # (1, 1, 12, 16)
        R_yaw = self._build_R_yaw(R_body)                # (3, 3)
        state = self._build_state(v_world, R_yaw, R_body, p_current)
        act, _, self.hx = self.model(x, state, self.hx)  # act: (1, 6)
        thrust_cmd = self._postprocess_action(act, R_yaw) # (3,)

        # ── 后处理：加速度 → 世界系速度 → 体坐标系速度 ──
        v_cmd_world = v_world + thrust_cmd * self.control_dt  # 加速度积分
        v_cmd_body = R_body.T @ v_cmd_world  # 转到机体坐标系

        # ── 9. 发布指令 ──
        self._publish_cmd(v_cmd_body)

        # ── 10. 周期日志 ──
        self._step_count += 1
        if self._step_count % 45 == 0:  # 每 3 秒打印一次
            dist = (self.target - p_current).norm().item()
            speed = v_world.norm().item()
            self.get_logger().info(
                f"[model] [step {self._step_count}] "
                f"pos=({p_current[0]:.2f}, {p_current[1]:.2f}, {p_current[2]:.2f}) "
                f"dist={dist:.2f}m speed={speed:.2f}m/s "
                f"acc=({thrust_cmd[0]:.2f}, {thrust_cmd[1]:.2f}, {thrust_cmd[2]:.2f}) "
                f"v_body=({v_cmd_body[0]:.2f}, {v_cmd_body[1]:.2f}, {v_cmd_body[2]:.2f})"
            )

    # ── 数据解析辅助方法 ─────────────────────────────────────

    def _parse_depth(self, msg: Image) -> torch.Tensor:
        """解析 ROS Image (32FC1) → (H, W) torch tensor"""
        depth_np = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        # 处理 inf / nan
        depth_np = np.nan_to_num(depth_np, nan=25.0, posinf=25.0, neginf=0.1)
        return torch.from_numpy(depth_np.copy()).to(self.device)

    def _parse_odom(self, msg: Odometry):
        """解析 Odometry → (p, v_world, R_body)"""
        # 位置
        p = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=torch.float32, device=self.device)

        # 姿态 → 旋转矩阵
        q = msg.pose.pose.orientation
        R_np = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        R_body = torch.from_numpy(R_np).float().to(self.device)  # (3, 3)

        # 速度（body frame → world frame）
        v_body = torch.tensor([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=torch.float32, device=self.device)
        v_world = R_body @ v_body  # (3,)

        return p, v_world, R_body

    # ── 预处理 / 后处理 ─────────────────────────────────────

    def _preprocess_depth(self, depth_raw: torch.Tensor) -> torch.Tensor:
        """(H, W) 原始深度 → (1, 1, 12, 16) 视差特征图"""
        # Resize -> 48x64
        depth = F.interpolate(
            depth_raw[None, None], size=(48, 64),
            mode="bilinear", align_corners=False,
        )[0, 0]  # (48, 64)

        # 深度 → 视差变换
        x = 3.0 / depth.clamp(0.3, 24) - 0.6

        # 4x4 MaxPool
        x = F.max_pool2d(x[None, None], 4, 4)  # (1, 1, 12, 16)
        return x

    def _build_R_yaw(self, R_body: torch.Tensor) -> torch.Tensor:
        """从完整姿态矩阵构建 yaw-only 旋转矩阵 (3, 3)"""
        fwd = R_body[:, 0].clone()
        fwd[2] = 0.0
        fwd = F.normalize(fwd, dim=0)

        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        left = torch.linalg.cross(up, fwd)
        R_yaw = torch.stack([fwd, left, up], dim=-1)  # (3, 3)
        return R_yaw

    def _build_state(
        self, v_world: torch.Tensor, R_yaw: torch.Tensor,
        R_body: torch.Tensor, p_current: torch.Tensor,
    ) -> torch.Tensor:
        """构建 (1, 10) 状态向量"""
        # 航向坐标系下的当前速度
        local_v = v_world @ R_yaw  # (3,)

        # 目标速度（世界坐标系 → 航向坐标系）
        target_v_raw = self.target - p_current
        target_v_norm = target_v_raw.norm()
        if target_v_norm > 1e-6:
            target_v = (target_v_raw / target_v_norm) * min(target_v_norm.item(), self.max_speed)
        else:
            target_v = torch.zeros(3, device=self.device)
        target_v_local = target_v @ R_yaw  # (3,)

        # 机体上方向
        body_up = R_body[:, 2]  # (3,)

        # 碰撞半径
        margin = torch.tensor([self.margin], device=self.device)

        state = torch.cat([local_v, target_v_local, body_up, margin]).unsqueeze(0)  # (1, 10)
        return state

    def _postprocess_action(self, act: torch.Tensor, R_yaw: torch.Tensor) -> torch.Tensor:
        """模型输出 (1, 6) → 世界坐标系加速度指令 (3,)"""
        R_yaw_b = R_yaw.unsqueeze(0)  # (1, 3, 3)
        a_pred, v_pred = (R_yaw_b @ act.reshape(1, 3, 2)).unbind(-1)
        # a_pred: (1, 3), v_pred: (1, 3)

        # thrust_cmd = (a_pred - v_pred - g) * thr_est_error + g
        # thr_est_error = 1.0
        thrust_cmd = a_pred - v_pred  # 简化：(a - v - g)*1 + g = a - v
        return thrust_cmd.squeeze(0)  # (3,)

    # ── 发布指令 ─────────────────────────────────────────────

    def _publish_cmd(self, v_cmd_body: torch.Tensor):
        """发布机体坐标系下的速度指令"""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED  # 8

        # type_mask: 忽略位置、加速度、绝对航向，只使用速度 + yaw_rate
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
            | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW
        )

        msg.velocity.x = float(v_cmd_body[0])  # 前 (FLU X)
        msg.velocity.y = float(v_cmd_body[1])  # 左 (FLU Y)
        msg.velocity.z = float(v_cmd_body[2])  # 上 (FLU Z)
        msg.yaw_rate = 0.0

        self.pub_cmd.publish(msg)

    # ── 外部控制接口 ─────────────────────────────────────────

    def set_target(self, x: float, y: float, z: float):
        """动态更新目标位置并重置 GRU 隐状态"""
        self.target = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
        self.hx = None
        self.get_logger().info(f"目标已更新为 ({x:.2f}, {y:.2f}, {z:.2f})，GRU 已重置")


def main():
    parser = argparse.ArgumentParser(description="DiffPhy ROS2 Inference Node")
    parser.add_argument(
        "--checkpoint", type=str, default="/ws/checkpoint0004.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=[10.0, 0.0, 1.0],
        help="目标位置 [x, y, z]（世界坐标系 FLU，单位米）",
    )
    parser.add_argument(
        "--max_speed", type=float, default=2.0,
        help="目标速度上限 (m/s)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.15,
        help="碰撞安全半径 (m)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="计算设备 (cuda / cpu)",
    )

    # 兼容 ROS2 launch 传入的额外参数
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = InferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中断信号，节点退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
