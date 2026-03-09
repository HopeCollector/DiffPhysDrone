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
import csv
import io
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget

from scipy.spatial.transform import Rotation

# ────────────────────────── 导入模型 ──────────────────────────
from bitpilot.model import Model


class InferenceNode(Node):
    """基于训练模型的 15 Hz 推理节点"""

    def __init__(self, args):
        super().__init__("diffphy_inference")

        # ── 参数 ──────────────────────────────────────────────
        self.device = torch.device(args.device)
        self.max_speed = args.max_speed
        self.margin = args.margin
        self.control_dt = 1.0 / 15.0  # 控制周期 ~15 Hz
        # 目标位置：优先从 /uav1/target_pose 话题接收，--target 仅作初始值
        if args.target is not None:
            self.target = torch.tensor(args.target, dtype=torch.float32, device=self.device)
        else:
            self.target = None  # 等待话题输入
        self.target_yaw: float | None = None  # 从话题接收的目标航向（rad），None 时回退到位置朝向
        self._target_from_topic = False  # 标记是否已从话题接收过目标
        self.gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float32, device=self.device)

        # ── Yaw 跟随参数 ────────────────────────────────────
        # 训练中 yaw 跟随目标方向转动，yaw_inertia=5, yaw_ctl_delay≈6
        # 等效 yaw_rate 比例增益：普通 PID 近似
        self.yaw_rate_gain = args.yaw_rate_gain    # rad/s per rad of error
        self.yaw_rate_max = args.yaw_rate_max      # rad/s 最大角速度

        # ── 加载模型 ──
        self.model = Model(dim_obs=10, dim_action=6).to(self.device)
        state_dict = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.get_logger().info(f"模型加载完成: {args.checkpoint}")

        # ── GRU 隐状态 ───────────────────────────────────────
        self.hx: torch.Tensor | None = None

        # ── 诊断日志文件 ─────────────────────────────────────
        self._init_flight_log(args.log_dir)

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
        self.sub_target = self.create_subscription(
            PoseStamped, "/uav1/target_pose",
            self._target_cb, qos_reliable,
        )

        # ── 发布 ─────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(
            PositionTarget, "/uav1/setpoint_raw/local", qos_best_effort,
        )

        # ── 15 Hz 定时器 ─────────────────────────────────────
        self.timer = self.create_timer(self.control_dt, self._control_loop)
        self._step_count = 0

        if args.target is not None:
            self.get_logger().info(
                f"推理节点启动 | 初始目标: {args.target} | max_speed: {args.max_speed} m/s | margin: {args.margin} m"
            )
        else:
            self.get_logger().info(
                f"推理节点启动 | 等待目标输入 | max_speed: {args.max_speed} m/s | margin: {args.margin} m"
            )
        self.get_logger().info("订阅 /uav1/target_pose 话题接收目标位置（也可通过 --target 指定初始值）")

    # ── 回调：仅缓存最新数据 ─────────────────────────────────

    def _depth_cb(self, msg: Image):
        with self._lock:
            self._depth_msg = msg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom_msg = msg

    def _target_cb(self, msg: PoseStamped):
        """从话题更新目标位置和航向"""
        new_target = torch.tensor([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=torch.float32, device=self.device)
        # 从四元数提取 yaw
        q = msg.pose.orientation
        new_yaw = float(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2])
        with self._lock:
            self.target = new_target
            self.target_yaw = new_yaw
            if not self._target_from_topic:
                self._target_from_topic = True
                self.get_logger().info("已切换到话题目标输入模式 (/uav1/target_pose)")

    # ── 主控制循环 ────────────────────────────────────────────

    @torch.no_grad()
    def _control_loop(self):
        # ── 1. 取出缓存数据 ──
        with self._lock:
            depth_msg = self._depth_msg
            odom_msg = self._odom_msg

        if self.target is None:
            if self._step_count % 45 == 0:
                self.get_logger().warn("等待目标位置... (发布到 /uav1/target_pose 或使用 --target 参数)")
            self._step_count += 1
            return

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

        # ── 3. 深度图预处理（带统计） ──
        depth_raw = self._parse_depth(depth_msg)
        depth_min = depth_raw.min().item()
        depth_max = depth_raw.max().item()
        depth_mean = depth_raw.mean().item()

        x = self._preprocess_depth(depth_raw)            # (1, 1, 12, 16)
        disp_min = x.min().item()
        disp_max = x.max().item()
        disp_mean = x.mean().item()

        # ── 4. 构建输入并推理 ──
        R_yaw = self._build_R_yaw(R_body)                # (3, 3)
        state = self._build_state(v_world, R_yaw, R_body, p_current)
        act, _, self.hx = self.model(x, state, self.hx)  # act: (1, 6)
        a_pred, v_pred, thrust_cmd = self._postprocess_action(act, R_yaw)

        # ── 5. 加速度 → 世界系速度 → 体坐标系速度 ──
        v_cmd_world = v_world + thrust_cmd * self.control_dt  # 加速度积分
        v_cmd_body = R_body.T @ v_cmd_world  # 转到机体坐标系

        # ── 5b. 计算 yaw_rate（机头跟随目标方向） ──
        yaw_rate = self._compute_yaw_rate(p_current, R_body)

        # ── 6. 发布指令 ──
        self._publish_cmd(v_cmd_body, yaw_rate)

        # ── 7. 诊断数据（每步写文件） ──
        self._step_count += 1
        t_now = time.monotonic()
        dist = (self.target - p_current).norm().item()
        speed = v_world.norm().item()
        hx_norm = self.hx.norm().item() if self.hx is not None else 0.0

        # 姿态角（从旋转矩阵 → RPY）
        euler = Rotation.from_matrix(R_body.cpu().numpy()).as_euler("xyz", degrees=True)
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])

        # 航向 vs 目标方位
        target_dir = self.target - p_current
        target_bearing = math.degrees(math.atan2(target_dir[1].item(), target_dir[0].item()))
        yaw_error = target_bearing - yaw
        # 归一化到 [-180, 180]
        yaw_error = (yaw_error + 180) % 360 - 180

        # state 向量各分量
        s = state.squeeze(0)  # (10,)
        local_v = s[0:3]
        target_v_local = s[3:6]
        body_up = s[6:9]

        self._write_flight_log(
            step=self._step_count, t=t_now,
            # 位姿
            px=p_current[0].item(), py=p_current[1].item(), pz=p_current[2].item(),
            roll=roll, pitch=pitch, yaw=yaw,
            # 速度
            vw_x=v_world[0].item(), vw_y=v_world[1].item(), vw_z=v_world[2].item(),
            speed=speed,
            # 距离/航向
            dist=dist, target_bearing=target_bearing, yaw_error=yaw_error,
            # 模型输出
            a_pred_x=a_pred[0].item(), a_pred_y=a_pred[1].item(), a_pred_z=a_pred[2].item(),
            v_pred_x=v_pred[0].item(), v_pred_y=v_pred[1].item(), v_pred_z=v_pred[2].item(),
            thrust_x=thrust_cmd[0].item(), thrust_y=thrust_cmd[1].item(), thrust_z=thrust_cmd[2].item(),
            # 指令
            vcmd_bx=v_cmd_body[0].item(), vcmd_by=v_cmd_body[1].item(), vcmd_bz=v_cmd_body[2].item(),
            yaw_rate_cmd=yaw_rate,
            # 深度图统计
            depth_min=depth_min, depth_max=depth_max, depth_mean=depth_mean,
            disp_min=disp_min, disp_max=disp_max, disp_mean=disp_mean,
            # state 向量
            lv_x=local_v[0].item(), lv_y=local_v[1].item(), lv_z=local_v[2].item(),
            tv_x=target_v_local[0].item(), tv_y=target_v_local[1].item(), tv_z=target_v_local[2].item(),
            bup_x=body_up[0].item(), bup_y=body_up[1].item(), bup_z=body_up[2].item(),
            # GRU
            hx_norm=hx_norm,
        )

        # ── 8. 终端日志（低频，每 3 秒） ──
        if self._step_count % 45 == 0:
            self.get_logger().info(
                f"[step {self._step_count}] "
                f"pos=({p_current[0]:.2f}, {p_current[1]:.2f}, {p_current[2]:.2f}) "
                f"dist={dist:.1f}m spd={speed:.2f}m/s "
                f"yaw={yaw:.1f}° err={yaw_error:.1f}° "
                f"hx={hx_norm:.1f} "
                f"depth=[{depth_min:.1f},{depth_mean:.1f},{depth_max:.1f}] "
                f"acc=({thrust_cmd[0]:.2f},{thrust_cmd[1]:.2f},{thrust_cmd[2]:.2f}) "
                f"v_body=({v_cmd_body[0]:.2f},{v_cmd_body[1]:.2f},{v_cmd_body[2]:.2f})"
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

    def _postprocess_action(self, act: torch.Tensor, R_yaw: torch.Tensor):
        """模型输出 (1, 6) → (a_pred, v_pred, thrust_cmd)，均为 (3,)"""
        R_yaw_b = R_yaw.unsqueeze(0)  # (1, 3, 3)
        a_pred, v_pred = (R_yaw_b @ act.reshape(1, 3, 2)).unbind(-1)
        # a_pred: (1, 3), v_pred: (1, 3)

        # thrust_cmd = (a_pred - v_pred - g) * thr_est_error + g
        # thr_est_error = 1.0
        thrust_cmd = a_pred - v_pred  # 简化：(a - v - g)*1 + g = a - v
        return a_pred.squeeze(0), v_pred.squeeze(0), thrust_cmd.squeeze(0)

    def _compute_yaw_rate(self, p_current: torch.Tensor, R_body: torch.Tensor) -> float:
        """
        计算 yaw_rate 使机头跟随目标方向。

        训练中 yaw 由 update_state_vec 控制：
            forward = current_fwd * yaw_inertia(5) + target_direction
            forward = current_fwd * alpha + normalize(forward) * (1-alpha)
        机头缓慢转向目标位置方向。这里用简单的 P 控制器近似。
        """
        # 如果有从话题接收的目标航向，使用它；否则从目标位置方向计算
        if self.target_yaw is not None:
            target_yaw = self.target_yaw
        else:
            target_dir = self.target - p_current
            target_yaw = math.atan2(target_dir[1].item(), target_dir[0].item())

        # 当前机头方向（从旋转矩阵提取）
        fwd = R_body[:, 0]  # 机体 X 轴
        current_yaw = math.atan2(fwd[1].item(), fwd[0].item())

        # 角度差（归一化到 [-pi, pi]）
        yaw_err = target_yaw - current_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        # P 控制  + clamp
        yaw_rate = self.yaw_rate_gain * yaw_err
        yaw_rate = max(-self.yaw_rate_max, min(self.yaw_rate_max, yaw_rate))

        return yaw_rate

    # ── 发布指令 ─────────────────────────────────────────────

    def _publish_cmd(self, v_cmd_body: torch.Tensor, yaw_rate: float = 0.0):
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
        msg.yaw_rate = float(yaw_rate)

        self.pub_cmd.publish(msg)

    # ── 外部控制接口 ─────────────────────────────────────────

    def set_target(self, x: float, y: float, z: float):
        """动态更新目标位置并重置 GRU 隐状态"""
        self.target = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
        self.hx = None
        self.get_logger().info(f"目标已更新为 ({x:.2f}, {y:.2f}, {z:.2f})，GRU 已重置")

    # ── 飞行诊断日志（CSV 文件） ──────────────────────────────

    _LOG_COLUMNS = [
        "step", "t",
        # 位姿
        "px", "py", "pz", "roll", "pitch", "yaw",
        # 速度
        "vw_x", "vw_y", "vw_z", "speed",
        # 距离/航向
        "dist", "target_bearing", "yaw_error",
        # 模型输出拆分
        "a_pred_x", "a_pred_y", "a_pred_z",
        "v_pred_x", "v_pred_y", "v_pred_z",
        "thrust_x", "thrust_y", "thrust_z",
        # 下发的机体系速度指令
        "vcmd_bx", "vcmd_by", "vcmd_bz",
        "yaw_rate_cmd",
        # 深度图统计
        "depth_min", "depth_max", "depth_mean",
        "disp_min", "disp_max", "disp_mean",
        # state 向量分量
        "lv_x", "lv_y", "lv_z",
        "tv_x", "tv_y", "tv_z",
        "bup_x", "bup_y", "bup_z",
        # GRU
        "hx_norm",
    ]

    def _init_flight_log(self, log_dir: str):
        """初始化 CSV 飞行日志文件"""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = log_path / f"flight_{ts}.csv"
        self._log_file = open(self._log_file_path, "w", newline="")
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=self._LOG_COLUMNS)
        self._log_writer.writeheader()
        self._log_flush_counter = 0
        self.get_logger().info(f"飞行日志: {self._log_file_path}")

    def _write_flight_log(self, **kwargs):
        """写入一行 CSV 诊断数据（每控制步调用一次）"""
        # 四舍五入浮点数减小文件体积
        row = {k: round(v, 4) if isinstance(v, float) else v for k, v in kwargs.items()}
        self._log_writer.writerow(row)
        self._log_flush_counter += 1
        if self._log_flush_counter >= 30:  # 每 2 秒 flush 一次
            self._log_file.flush()
            self._log_flush_counter = 0

    def destroy_node(self):
        """关闭日志文件后销毁节点"""
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self.get_logger().info(f"飞行日志已保存: {self._log_file_path}")
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="DiffPhy ROS2 Inference Node")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/checkpoint0004.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=None,
        help="初始目标位置 [x, y, z]（世界坐标系 FLU，单位米）。不指定时等待 /uav1/target_pose 话题",
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
        "--yaw_rate_gain", type=float, default=1.5,
        help="Yaw 跟随 P 增益 (rad/s per rad)",
    )
    parser.add_argument(
        "--yaw_rate_max", type=float, default=1.5,
        help="最大 yaw_rate (rad/s)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="计算设备 (cuda / cpu)",
    )
    parser.add_argument(
        "--log_dir", type=str, default="/ws/flight_logs",
        help="飞行诊断日志目录",
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
