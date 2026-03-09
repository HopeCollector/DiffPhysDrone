---
name: sim-control
description: >
  Control the Gazebo UAV simulation lifecycle via HTTP API at gz-sim:6799.
  Use this skill when asked to start, stop, pause, resume, restart the simulation,
  check simulation status, or switch simulation configs (single, single_camera_only, swarm).
argument-hint: "[start|stop|pause|resume|restart|status] [config]"
---

# Gazebo Simulation Control

Control the UAV simulation running on the `gz-sim` host via its HTTP API.

## Server

- **Base URL**: `http://gz-sim:6799`
- **Foxglove WebSocket**: `ws://gz-sim:8765` (auto-managed with the simulation)

## Endpoints

| Method | Path       | Body (JSON, optional)    | Description          |
|--------|------------|--------------------------|----------------------|
| GET    | `/`        | —                        | Help / API reference |
| GET    | `/status`  | —                        | Current state        |
| POST   | `/start`   | `{"config": "<name>"}`   | Start simulation     |
| POST   | `/stop`    | —                        | Stop simulation      |
| POST   | `/pause`   | —                        | Pause physics        |
| POST   | `/resume`  | —                        | Resume physics       |
| POST   | `/restart` | `{"config": "<name>"}`   | Stop + Start         |

## Config 参数

`config` 字段接受 **相对路径**，指向仿真服务器上的 YAML 配置文件，例如：

```
share/gz/diffphy-single.yaml
```

常用配置文件：

| 路径                                    | Description                   |
|-----------------------------------------|-------------------------------|
| `share/gz/diffphy-single.yaml`          | 单机仿真（默认，带障碍物）     |
| `share/gz/diffphy-single_camera_only.yaml` | 单机仿真（仅相机，无障碍物）|
| `share/gz/diffphy-swarm.yaml`           | 多机集群仿真                   |

> **注意**: 下方示例中出现的 `"config":"single"` 等短名称仅为演示用途，实际使用时请传入完整的相对路径（如 `share/gz/diffphy-single.yaml`）。

If body is omitted, the last-used config is used.

## Response Format

All endpoints return JSON. Key fields:

```json
{
  "state": "running|paused|stopped",
  "config": "single",
  "message": "仿真已启动 (config=single, pid=1919)",
  "sim_pid": 1919,
  "foxglove_pid": 1920,
  "started_at": "2026-03-05T07:01:19.479667+00:00",
  "uptime_s": 5.0
}
```

`state` is always present. Other fields depend on the endpoint and current state.

## How To Use

All commands use `curl` from within this dev container. The hostname `gz-sim` resolves to the simulation server container.

### Check Status

```bash
curl -s gz-sim:6799/status
```

### Start Simulation

```bash
# Start with specific config (传入相对路径)
curl -s -X POST gz-sim:6799/start -H 'Content-Type: application/json' -d '{"config":"share/gz/diffphy-single.yaml"}'

# Start with last-used config
curl -s -X POST gz-sim:6799/start
```

### Stop Simulation

```bash
curl -s -X POST gz-sim:6799/stop
```

### Pause / Resume

```bash
curl -s -X POST gz-sim:6799/pause
curl -s -X POST gz-sim:6799/resume
```

### Restart (stop + start)

```bash
curl -s -X POST gz-sim:6799/restart -H 'Content-Type: application/json' -d '{"config":"share/gz/diffphy-single.yaml"}'
```

## Important Notes

- Always check `/status` before starting — starting when already running will fail.
- After `/start` or `/restart`, wait 3-5 seconds for ROS2 topics to become available.
- Use `/restart` instead of manual stop+start for config changes.
- The simulation must be running for any ROS2 topics (depth, odometry, IMU) to publish data.
