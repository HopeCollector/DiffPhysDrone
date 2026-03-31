# Project Guidelines

## Overview

BitPilot is a vision-based agile flight training and deployment framework built on differentiable physics simulation. It serves as the single-drone safety control core for the skybit UAV swarm project. The system trains a CNN+GRU neural network end-to-end through a differentiable simulator, then deploys via ROS2 inference nodes to Gazebo simulation or physical hardware (NVIDIA Orin NX 16G).

## Environment

- Ubuntu 24.04 LTS (dev container)
- Python 3.12.3
- uv 0.9.18 (package manager)
- CUDA 12.9 / nvcc 12.9
- PyTorch 2.8+ with CUDA 12.8
- ROS2 Jazzy (deployment only)
- Git remote: `origin https://github.com/HopeCollector/DiffPhysDrone.git` / branch: `dev`

## Architecture

```
Core Pipeline: Differentiable Sim (bitpilot._C + env.py) → Training (train.py) → ROS2 Deploy (inference_node.py) → Gazebo/Hardware

src/bitpilot/
├── _csrc/                 # C++/CUDA extensions → compiled as bitpilot._C
│   ├── quadsim.cpp        #   pybind11 bindings (render, find_nearest_pt, rerender_backward)
│   └── quadsim_kernel.cu  #   Render & collision detection CUDA kernels
├── model.py               # CNN+GRU network (disparity→4-layer CNN→GRU→6-dim action)
├── env.py                 # Differentiable sim environment (~420 lines, dynamics in pure PyTorch)
├── train.py               # Training loop (barrier loss, AdamW, CosineAnnealingLR)
└── deploy/
    ├── inference_node.py  # ROS2 inference node (depth+odom → velocity cmd)
    └── joy_target_node.py # Joystick → target pose node
```

Full architecture details: see `docs/CONTEXT.md`.

## Documentation Roles

- `docs/CONTEXT.md` is the current-state snapshot, not a chronological ledger. Update it in place only when the best concise description of the project's present state has changed.
- `docs/CONTEXT.md` should stay compressible and rewriteable. Prefer replacing outdated sections over appending incremental notes.
- `docs/DEV.md` is the append-only record for decisions, rationale, and verification that should remain historically visible. Append to it once per resolved problem or meaningful milestone, not once per edit or per conversation turn.
- Not every code or document change requires a `docs/CONTEXT.md` edit. First assess whether the change materially affects the current project snapshot.
- If a change only matters as history, discussion, or implementation detail, record it in `docs/DEV.md` or leave it out of documentation entirely.
- `refs/REF.md` is curated reference knowledge, not project progress tracking. Update it only when durable, reusable knowledge has changed.
- Unless the user explicitly asks for direct manual edits, updates to `docs/CONTEXT.md`, `docs/DEV.md`, `.github/copilot-instructions.md`, and affected `README.md` files should be handled through the user-level `update-docs-workflow` skill in a fresh subagent after the problem is solved or a meaningful milestone is complete.

## Key Files

Read these **on demand** — not all at once.

| File | Purpose | When to Read |
|------|---------|--------------|
| `docs/CONTEXT.md` | Current-state snapshot: structure, stack, pitfalls, status. Update in place, compress when needed | Start of every new conversation |
| `docs/DEV.md` | Development log (append-only) | Only when historical context is needed |
| `docs/plans/*.plan.md` | Feature development plans | When working on a planned feature |
| `docs/plans/README.md` | Plan authoring conventions | Before creating a new plan |
| `refs/REF.md` | Curated notes from dependency docs | When implementing integrations |
| `refs/README.md` | Reference doc conventions | Before gathering new reference material |

## Development Workflow

For every change request, follow this sequence:

1. **Clarify** — Confirm requirements and acceptance criteria
2. **Analyze impact** — Identify affected files, flag breaking changes
3. **Write tests first** — Build test cases from requirements (TDD when applicable)
4. **Implement** — Minimal, focused changes only
5. **Unit test** — Verify the change works
6. **Full test suite** — Ensure no regressions
7. **Security & performance check** — When the change touches trust boundaries or hot paths
8. **Update docs** — After the problem is solved or a meaningful milestone is complete, run the user-level `update-docs-workflow` skill if needed; rewrite `docs/CONTEXT.md` only when the current-state snapshot changed materially, and append to `docs/DEV.md` only once for durable historical context
9. **Update README** — If user-facing behavior changed
10. **Report** — Summarize what changed, what was tested, any concerns
11. **Commit** — After user confirmation, use conventional commit format

## Project-Specific Notes

- Coordinate system: **FLU** (Front-Left-Up) throughout the entire pipeline. No coordinate transforms needed between sim and Gazebo.
- CUDA extension `bitpilot._C` uses **positional-only** arguments (pybind11, no kwargs).
- Dynamics are pure PyTorch in `env.py`; CUDA extension only handles render/collision (`render`, `find_nearest_pt`, `rerender_backward`).
- Build: `uv sync` compiles everything. `no-build-isolation-package = ["bitpilot"]` ensures ABI compatibility with venv torch.
- Control frequency: 15 Hz.
- Tests requiring Gazebo are marked `@pytest.mark.preflight` or `@pytest.mark.flight`; GPU tests are marked `@pytest.mark.gpu`.
