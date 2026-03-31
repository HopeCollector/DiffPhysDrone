# Project Context

> Living snapshot — update in place, never append. Must always reflect current state.

## Overview

BitPilot is a differentiable-physics-based vision agile flight framework. It trains a CNN+GRU policy end-to-end through a fully differentiable quadrotor simulator, then deploys via ROS2 to Gazebo or physical drones (target: NVIDIA Orin NX 16G). Part of the skybit swarm project — this repo handles single-drone obstacle avoidance. Current maturity: **alpha** (training works, sim deployment works, improvement plan in progress).

## Directory Structure

```
src/bitpilot/
├── __init__.py                # Package init, version 0.1.0
├── env.py                     # Differentiable sim environment (Env class, ~420 lines)
│                              #   Pure PyTorch dynamics + CUDA render/collision
├── model.py                   # CNN+GRU network: disparity(B,1,12,16) + state(B,10) → action(B,6)
├── train.py                   # Training loop: barrier loss, AdamW, CosineAnnealingLR
├── _csrc/                     # C++/CUDA source → compiled as bitpilot._C
│   ├── quadsim.cpp            #   pybind11 bindings
│   └── quadsim_kernel.cu      #   Render & collision CUDA kernels
├── _C.cpython-312-*.so        # Compiled extension binary
└── deploy/
    ├── inference_node.py      # ROS2 node: depth+odom → velocity commands (517 lines)
    └── joy_target_node.py     # ROS2 node: joystick → target pose

configs/
├── single_agent.args          # Single drone: speed_mtp=4, cam_angle=20, fov=0.82, obstacles
└── multi_agent.args           # Multi drone: gate mode, batch=256, timesteps=180

tests/
├── test_model.py              # Model unit tests (7 tests) ✅
├── test_env.py                # Sim environment tests (10 tests) ✅
├── test_cuda_gradient.py      # Gradient verification (2 tests) ✅
├── test_cuda_dynamics_forward.py   # Forward dynamics (12 tests) ✅
├── test_cuda_dynamics_backward.py  # Backward dynamics (3 tests) ✅
├── test_cuda_collision.py     # Collision/state update (14 tests) ✅
├── test_cuda_render.py        # Render kernel (14 tests) ✅
├── test_preflight.py          # Gazebo preflight checks (needs running sim)
└── test_flight.py             # Flight integration tests (needs running sim)

checkpoints/                   # Training snapshots: checkpoint0000–0004.pth
runs/                          # TensorBoard event logs
flight_logs/                   # CSV flight recordings from sim deployment
scripts/run_inference.sh       # Inference launcher (sources ROS2 + venv)
check_sim.py                   # Preflight check wrapper (delegates to pytest)
```

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| PyTorch | ≥2.8.0 (with CUDA 12.8) |
| CUDA Toolkit | 12.9 |
| uv (pkg mgr) | 0.9.18 |
| ROS2 | Jazzy (deployment only) |
| OS | Ubuntu 24.04 LTS (dev container) |
| Build | setuptools + CUDAExtension (via setup.py), metadata in pyproject.toml |

Key dependencies: numpy, opencv-python, scipy, tensorboard, aim, transforms3d, tqdm, matplotlib, pyyaml, lark.

## Known Pitfalls

**CUDA extension ABI mismatch** → PEP 517 build isolation downloads a separate torch with incompatible ABI → `ImportError: undefined symbol`. Fix: `no-build-isolation-package = ["bitpilot"]` in pyproject.toml ensures `uv sync` uses venv's torch.

**bitpilot._C positional-only args** → pybind11 bindings don't support keyword arguments. Always call `render(a, b, c)`, never `render(a=a, b=b)`.

**First-time build takes ~2 min** → CUDA kernel compilation. Subsequent `uv sync` skips if sources unchanged.

## Current Status

- Training pipeline functional for single-agent (obstacle avoidance) and multi-agent (gate passing) modes
- 5 checkpoints saved (checkpoint0000–0004)
- ROS2 inference deployment working with Gazebo
- 62+ unit tests passing (model, env, CUDA kernels, gradients)
- **Planned improvement**: speed schedule + hover training (see `NO-USE/plan.md` for legacy plan, to be migrated to `docs/plans/`)
- Known issue: model lacks deceleration/hover behavior — oscillates 1m↔3.7m near target
- Project infrastructure being reorganized (this initialization)
