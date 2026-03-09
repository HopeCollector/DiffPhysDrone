import math
import random
import time
import torch
import torch.nn.functional as F
import bitpilot._C as quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


def run_forward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
                act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt):
    """纯 PyTorch 动力学前向传播，autograd 自动处理反向传播。"""
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha

    # 体轴速度分量
    v_fwd_s, v_left_s, v_up_s = ((v - v_wind)[:, None] @ R).unbind(-1)

    # 二次 + 一次阻力
    drag = drag_2[:, :1] * (
        v_fwd_s.abs() * v_fwd_s * R[..., 0]
        + v_left_s.abs() * v_left_s * R[..., 1]
        + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef
    )
    drag = drag + drag_2[:, 1:] * (
        v_fwd_s * R[..., 0]
        + v_left_s * R[..., 1]
        + v_up_s * R[..., 2] * z_drag_coef
    )

    a_next = act_next + dg - drag

    # Verlet 积分 + g_decay（前向不变，反向缩放梯度）
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt ** 2
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt

    return act_next, p_next, v_next, a_next


run = run_forward


class Env:
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        """
        初始化四旋翼飞行模拟环境。
        
        参数:
            batch_size: 批次大小，同时模拟的无人机数量
            width: 渲染图像宽度（像素）
            height: 渲染图像高度（像素）
            grad_decay: 梯度衰减系数，用于反向传播时的梯度缩放
            device: 计算设备（'cpu' 或 'cuda'）
            fov_x_half_tan: 相机水平视场角的一半的正切值，默认约30度
            single: 是否为单机模式（不考虑多机编队）
            gate: 是否生成门框障碍物
            ground_voxels: 是否生成地面平面障碍物
            scaffold: 是否生成脚手架结构障碍物
            speed_mtp: 速度倍数，用于调整最大速度
            random_rotation: 是否随机旋转整个场景
            cam_angle: 相机俯仰角（度），用于调整机载相机视角
        """
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        
        # 环境障碍物参数：_w(Weight)为随机生成时的缩放范围，_b(Bias)为位置偏置
        # 球形障碍物：[x范围, y范围, z范围, 半径范围]
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        
        # 长方体障碍物：[x, y, z, x尺寸, y尺寸, z尺寸]
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        
        # 地面长方体障碍物：用于生成地面平面和起伏
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        
        # 垂直圆柱体障碍物：[x, y, 半径]
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        
        # 水平圆柱体障碍物：用于脚手架等结构
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        
        # 门框参数：[x位置, y位置, z位置, 半径]
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        
        # 风速扰动范围：[x方向, y方向, z方向]
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        
        # 标准重力加速度向量（m/s²）
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        
        # 屋顶障碍物附加参数：用于生成天花板和屋顶结构
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        
        # 子步长时间分割：将一个控制周期(1/15秒)分成10个子步进行物理仿真
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        
        # 无人机初始位置（米）：8个预设起点，支持多机编队
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],  # 左下
            [ 9.5, -3.,  1],  # 右下
            [-0.5,  1.,  1],  # 左中
            [ 8.5,  1.,  1],  # 右中
            [ 0.0,  3.,  1],  # 左上
            [ 8.0,  3.,  1],  # 右上
            [-1.0, -1.,  1],  # 左中下
            [ 9.0, -1.,  1],  # 右中下
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 无人机目标位置（米）：与初始位置对应的终点
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],   # 目标：右上
            [0.,  3.,  1],   # 目标：左上
            [8., -1.,  1],   # 目标：右中下
            [0., -1.,  1],   # 目标：左中下
            [8., -3.,  1],   # 目标：右下
            [0., -3.,  1],   # 目标：左下
            [8.,  1.,  1],   # 目标：右中
            [0.,  1.,  1],   # 目标：左中
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 光流缓冲区：用于渲染运动光流（当前为空）
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        
        # 环境配置标志
        self.single = single                    # 单机模式标志
        self.gate = gate                        # 是否启用门框
        self.ground_voxels = ground_voxels      # 是否启用地面障碍物
        self.scaffold = scaffold                # 是否启用脚手架
        self.speed_mtp = speed_mtp              # 速度倍增系数
        self.random_rotation = random_rotation  # 是否随机旋转场景
        self.cam_angle = cam_angle              # 相机俯仰角
        self.fov_x_half_tan = fov_x_half_tan    # 相机视场角参数
        
        # 初始化环境状态
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # scaffold
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = Env.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        GRAVITY = 9.80665
        g_offset = torch.tensor([0.0, 0.0, GRAVITY], device=R.device, dtype=R.dtype)
        a_thr_g = a_thr + g_offset

        raw_thrust = a_thr_g.norm(2, -1, keepdim=True)
        default_up = torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype).expand_as(a_thr_g)
        up = torch.where(raw_thrust < 1e-8, default_up, a_thr_g / raw_thrust.clamp(min=1e-8))

        fwd_old = R[..., 0]
        fwd = fwd_old * yaw_inertia + v_pred
        fwd = F.normalize(fwd, 2, -1)
        fwd = (1 - alpha) * fwd + alpha * fwd_old

        uz_safe = torch.where(
            up[:, 2:3] >= 0,
            up[:, 2:3].clamp(min=1e-6),
            up[:, 2:3].clamp(max=-1e-6),
        )
        fwd_z = (fwd[:, 0:1] * up[:, 0:1] + fwd[:, 1:2] * up[:, 1:2]) / (-uz_safe)
        fwd = torch.cat([fwd[:, 0:1], fwd[:, 1:2], fwd_z], dim=-1)
        fwd = F.normalize(fwd, 2, -1)

        left = torch.cross(up, fwd, dim=-1)
        return torch.stack([fwd, left, up], dim=-1)

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        """
        计算从无人机未来位置到最近障碍物表面点的向量。
        
        该方法用于计算无人机在下一个子步长位置处，到周围最近障碍物表面的向量。
        这个向量可用于碰撞检测、避障规划和安全距离计算。
        
        计算流程：
        1. 根据当前位置(p)和速度(v)，预测下一个子步长后的位置
        2. 调用CUDA内核查找该位置到所有障碍物的最近点
        3. 返回从预测位置指向最近障碍物点的向量
        
        障碍物类型包括：
        - balls: 球形障碍物（如无人机之间的碰撞检测）
        - cyl: 垂直圆柱体障碍物
        - cyl_h: 水平圆柱体障碍物
        - voxels: 长方体障碍物（如建筑物、地面、天花板）
        
        Returns:
            torch.Tensor: 形状为(B, 3)的张量，表示从无人机预测位置到最近障碍物点的向量
                         - 向量长度表示到障碍物的距离
                         - 向量方向指向障碍物
                         - 当无人机接近障碍物时，向量长度变小
        
        Note:
            - 该方法在训练中用于计算碰撞损失和避障损失
            - 使用CUDA加速计算，支持批量并行处理
            - 考虑了无人机半径(drone_radius)以确保安全间距
            - n_drones_per_group参数用于多机协同时的碰撞检测分组
        """
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = Env.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = Env.update_state_vec(self.R, self.act, v_pred, alpha, 5)

