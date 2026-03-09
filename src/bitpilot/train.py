from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from bitpilot.env import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from aim import Run

import argparse
from bitpilot.model import Model


def barrier(x: torch.Tensor, v_to_pt):
    """
    计算障碍函数(Barrier Function)的损失值。

    该函数实现了一个基于ReLU的二次障碍函数,用于约束优化问题中。
    当输入值x小于1时会产生惩罚,确保优化过程中满足特定约束条件。

    Args:
        x (torch.Tensor): 输入张量,通常表示需要约束的变量值。
                          期望值应该大于等于1,小于1的部分会被惩罚。
        v_to_pt: 权重系数,用于调节障碍函数的强度。可以是标量或张量,
                 用于控制不同约束的重要程度。

    Returns:
        torch.Tensor: 标量张量,表示障碍函数的平均损失值。
                      计算公式: mean(v_to_pt * relu(1-x)^2)

    工作原理:
        1. 计算 (1 - x):得到与目标值1的偏差
        2. 应用 relu():只保留负偏差(即x < 1的情况),将x >= 1的部分置零
        3. 求平方 pow(2):对违反约束的程度进行二次惩罚
        4. 乘以权重 v_to_pt:应用自定义的惩罚强度
        5. 求均值 mean():返回所有元素的平均障碍损失

    注意:
        - 当x >= 1时,relu(1-x) = 0,不产生任何惩罚
        - 当x < 1时,惩罚值随着偏差增大而二次增长
        - 这种形式常用于interior point方法和约束优化算法中
    """
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
    parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
    parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
    parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
    parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_decay', type=float, default=0.4)
    parser.add_argument('--speed_mtp', type=float, default=1.0)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
    parser.add_argument('--timesteps', type=int, default=150)
    parser.add_argument('--cam_angle', type=int, default=10)
    parser.add_argument('--single', default=False, action='store_true')
    parser.add_argument('--gate', default=False, action='store_true')
    parser.add_argument('--ground_voxels', default=False, action='store_true')
    parser.add_argument('--scaffold', default=False, action='store_true')
    parser.add_argument('--random_rotation', default=False, action='store_true')
    parser.add_argument('--yaw_drift', default=False, action='store_true')
    parser.add_argument('--no_odom', default=False, action='store_true')
    args = parser.parse_args()
    writer = SummaryWriter()
    aim_run = Run(repo='share/aim_data')
    aim_run['hparams'] = vars(args)
    print(args)

    device = torch.device('cuda')

    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              gate=args.gate, ground_voxels=args.ground_voxels,
              scaffold=args.scaffold, speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation, cam_angle=args.cam_angle)
    if args.no_odom:
        model = Model(7, 6)
    else:
        model = Model(7+3, 6)
    model = model.to(device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        if missing_keys:
            print("missing_keys:", missing_keys)
        if unexpected_keys:
            print("unexpected_keys:", unexpected_keys)
    optim = AdamW(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

    ctl_dt = 1 / 15


    scaler_q = defaultdict(list)
    def smooth_dict(ori_dict):
        for k, v in ori_dict.items():
            scaler_q[k].append(float(v))

    pbar = tqdm(range(args.num_iters), ncols=80)
    # depths = []
    # states = []
    B = args.batch_size
    for i in pbar:
        env.reset()
        model.reset()
        p_history = []
        v_history = []
        target_v_history = []
        vec_to_pt_history = []
        act_diff_history = []
        v_preds = []
        vid = []
        v_net_feats = []
        h = None

        act_lag = 1
        act_buffer = [env.act] * (act_lag + 1)
        target_v_raw = env.p_target - env.p
        if args.yaw_drift:
            drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            R_drift = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ], -1).reshape(B, 3, 3)


        for t in range(args.timesteps):
            # 采样控制时间步长，模拟控制时间步长的随机性 10% 的标准差
            ctl_dt = normalvariate(1 / 15, 0.1 / 15)
            # 运行环境一步，得到深度图。depth: (B, 12, 16)；flow：None
            depth, flow = env.render(ctl_dt)
            # 收集无人机当前位置
            p_history.append(env.p)
            # 收集无人机到最近障碍物的向量
            vec_to_pt_history.append(env.find_vec_to_nearest_pt())

            # 如果是保存迭代（比如每 1000 次），就把第 4 个无人机的深度图存下来，稍后生成视频或图片
            if is_save_iter(i):
                vid.append(depth[4])

            if args.yaw_drift:
                # 模拟航向漂移
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            else:
                # 直接使用目标位置减去当前位置，得到目标速度（未归一化）
                # .detach()：不计算梯度，防止梯度回传到环境模拟器
                target_v_raw = env.p_target - env.p.detach()
            # 仿真环境运行一步，使用上一次的动作
            env.run(act_buffer[t], ctl_dt, target_v_raw)

            # 无人机姿态（body -> world）
            R = env.R
            # 当前航向
            fwd = env.R[:, :, 0].clone()
            up = torch.zeros_like(fwd)
            # 把航向的 z 分量设为 0，得到水平投影
            fwd[:, 2] = 0
            # 上方向锁定为世界坐标系的 z 方向
            up[:, 2] = 1
            # 归一化航向向量
            fwd = F.normalize(fwd, 2, -1)
            # 重新计算正交基
            # 新的姿态矩阵仅航向与机体坐标系一致，但不考虑俯仰与横滚
            R = torch.stack([fwd, torch.cross(up, fwd), up], -1)
            # BUG：这样做会导致在俯仰角较大时，航向无法正确反映无人机实际前进方向，训练不出来能翻滚的策略

            # 限制目标速度
            target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v_unit = target_v_raw / target_v_norm
            target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

            # 构造观测状态 state
            state = [
                torch.squeeze(target_v[:, None] @ R, 1), # 1. 目标速度 (相对于机头方向)
                env.R[:, 2],                             # 2. 重力方向 (感知自己的倾斜姿态)
                env.margin[:, None]                      # 3. 自身半径 (我知道自己有多胖)
            ]

            # 计算当前速度相对于机头方向的表示
            local_v = torch.squeeze(env.v[:, None] @ R, 1)
            if not args.no_odom:                         # 4. 如果有里程计...
                state.insert(0, local_v)                 #    插入当前真实速度
            # 把所有状态量拼接在一起，得到最终的状态表示
            state = torch.cat(state, -1)

            # normalize
            # 计算视差图
            x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
            x = F.max_pool2d(x[:, None], 4, 4)

            # 神经网络前向传播
            act, values, h = model(x, state, h)

            # 把动作从本地坐标系转换到全局坐标系（俯仰与横滚不动）
            a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
            v_preds.append(v_pred)
            # 这里是在从期望加速度反推要给出多少加速度，直接看公式比较难受，反过来想：
            # 实际推力 = 期望推力 * 动力系统误差 + 与速度成正比的阻力 + 重力
            # 那么我们为了实现期望推力，需要给出多少实际推力就是
            # 给出推力 = (期望推力 - 重力 - 阻力) * 动力系统系数
            # 最后那个重力补偿来自 env.run，它会先做一次重力补偿
            act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            act_buffer.append(act)

            # 没用上
            v_net_feats.append(torch.cat([act, local_v, h], -1))

            v_history.append(env.v)
            target_v_history.append(target_v)

        p_history = torch.stack(p_history)
        loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
        act_buffer = torch.stack(act_buffer)

        v_history = torch.stack(v_history)
        # 返回一个同样形状的张量，其中第 i 行是前 i 行所有速度的总和
        v_history_cum = v_history.cumsum(0)
        v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
        target_v_history = torch.stack(target_v_history)
        T, B, _ = v_history.shape
        # 计算速度误差的 loss
        delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
        loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

        # 速度预测的 loss
        v_preds = torch.stack(v_preds)
        loss_v_pred = F.mse_loss(v_preds, v_history.detach())

        # 实际速度与目标速度的对齐 loss
        target_v_history_norm = torch.norm(target_v_history, 2, -1)
        target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
        # 实际速度方向在目标速度方向上的投影长度
        fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
        loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

        # 加加速度（急动度）
        jerk_history = act_buffer.diff(1, 0).mul(15)
        # 去掉重力干扰的纯加加加速度（跳动度）
        # TODO：虽然本意是计算姿态时不要出现除数为零的问题，但推力归零时仍然会出问题
        # 建议敢用加权平滑，当推力较小时，代价直接归零
        snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
        loss_d_acc = act_buffer.pow(2).sum(-1).mean()
        loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
        loss_d_snap = snap_history.pow(2).sum(-1).mean()

        vec_to_pt_history = torch.stack(vec_to_pt_history)
        distance = torch.norm(vec_to_pt_history, 2, -1)
        distance = distance - env.margin
        with torch.no_grad():
            # torch.diff(arr, n, dim)：计算张量 arr 在指定维度 dim 上的 n 阶离散差分
            # 计算的是接近障碍物的速度，所以前面有个负号表示接近障碍物时速度为正
            # ？？？ 135 是一个意义不明的超参数
            # BUG：diff 的计算维度应该是 0，因为 distance 的形状是 (T, B)
            v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
        # 当与障碍物距离小于 1m(在 barrier 函数中硬编码) 时，产生二次惩罚
        loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
        # 发生碰撞时会产生极大的惩罚
        loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

        speed_history = v_history.norm(2, -1)
        # 速度方向 loss
        loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

        loss = args.coef_v * loss_v + \
            args.coef_obj_avoidance * loss_obj_avoidance + \
            args.coef_bias * loss_bias + \
            args.coef_d_acc * loss_d_acc + \
            args.coef_d_jerk * loss_d_jerk + \
            args.coef_d_snap * loss_d_snap + \
            args.coef_speed * loss_speed + \
            args.coef_v_pred * loss_v_pred + \
            args.coef_collide * loss_collide + \
            args.coef_ground_affinity + loss_ground_affinity

        if torch.isnan(loss):
            print("loss is nan, exiting...")
            exit(1)

        pbar.set_description_str(f'loss: {loss:.3f}')
        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()


        with torch.no_grad():
            avg_speed = speed_history.mean(0)
            success = torch.all(distance.flatten(0, 1) > 0, 0)
            _success = success.sum() / B
            smooth_dict({
                'loss': loss,
                'loss_v': loss_v,
                'loss_v_pred': loss_v_pred,
                'loss_obj_avoidance': loss_obj_avoidance,
                'loss_d_acc': loss_d_acc,
                'loss_d_jerk': loss_d_jerk,
                'loss_d_snap': loss_d_snap,
                'loss_bias': loss_bias,
                'loss_speed': loss_speed,
                'loss_collide': loss_collide,
                'loss_ground_affinity': loss_ground_affinity,
                'success': _success,
                'max_speed': speed_history.max(0).values.mean(),
                'avg_speed': avg_speed.mean(),
                'ar': (success * avg_speed).mean()})
            log_dict = {}
            if is_save_iter(i):
                # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
                fig_p, ax = plt.subplots()
                p_history = p_history[:, 4].cpu()
                ax.plot(p_history[:, 0], label='x')
                ax.plot(p_history[:, 1], label='y')
                ax.plot(p_history[:, 2], label='z')
                ax.legend()
                fig_v, ax = plt.subplots()
                v_history = v_history[:, 4].cpu()
                ax.plot(v_history[:, 0], label='x')
                ax.plot(v_history[:, 1], label='y')
                ax.plot(v_history[:, 2], label='z')
                ax.legend()
                fig_a, ax = plt.subplots()
                act_buffer = act_buffer[:, 4].cpu()
                ax.plot(act_buffer[:, 0], label='x')
                ax.plot(act_buffer[:, 1], label='y')
                ax.plot(act_buffer[:, 2], label='z')
                ax.legend()
                # writer.add_video('demo', vid, i + 1, 15)
                writer.add_figure('p_history', fig_p, i + 1)
                writer.add_figure('v_history', fig_v, i + 1)
                writer.add_figure('a_reals', fig_a, i + 1)
            if (i + 1) % 10000 == 0:
                torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
            if (i + 1) % 25 == 0:
                for k, v in scaler_q.items():
                    avg_val = sum(v) / len(v)
                    writer.add_scalar(k, avg_val, i + 1)
                    aim_run.track(avg_val, name=k, step=i + 1)
                scaler_q.clear()

    aim_run.close()


if __name__ == "__main__":
    main()
