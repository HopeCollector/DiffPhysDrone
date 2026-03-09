#!/bin/bash
# ──────────────────────────────────────────────────────────────
# DiffPhy 推理节点启动脚本
#
# 用法:
#   ./run_inference.sh                           # 默认参数
#   ./run_inference.sh --target 10 0 1.5         # 自定义目标
#   ./run_inference.sh --max_speed 3.0           # 自定义速度
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 ROS2 环境
source /opt/ros/jazzy/setup.bash

# 激活项目虚拟环境（含 torch、scipy 等依赖）
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "──────────────────────────────────────"
echo "  BitPilot Inference Node"
echo "  Python: $(python3 --version 2>&1)"
echo "  Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "──────────────────────────────────────"

# 运行推理节点，透传所有命令行参数
exec python3 -m bitpilot.deploy.inference_node "$@"
