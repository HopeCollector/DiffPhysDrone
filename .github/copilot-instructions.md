# 项目环境与配置说明

> **新会话第一件事**：读取 `CONTEXT.md`，获取项目全貌、代码地图、已知陷阱。
> 本文件仅记录系统环境事实，不重复 CONTEXT.md 的内容。

这份文档记录了本项目的运行环境、依赖配置以及系统探测结果。

## 1. 项目概览

*   **名称**: BitPilot
*   **描述**: 基于可微物理（Differentiable Physics）的视觉敏捷飞行训练与部署框架。
*   **硬件适配**: 已针对 **NVIDIA RTX 5090** 进行适配。
*   **核心模块**:
    *   `bitpilot`: 主训练/部署框架（src-layout，源码在 `src/bitpilot/`）。
    *   `bitpilot._C`: CUDA 加速四旋翼动力学内核（C++/CUDA 扩展，源码在 `src/bitpilot/_csrc/`）。

## 2. 系统环境 (System Environment)

*   **操作系统**: Ubuntu 24.04.2 LTS (Noble Numbat)
*   **运行环境**: Docker 容器环境
*   **内核信息**: Linux (运行在容器内)

## 3. ROS2 环境

*   **版本 Distribution**: ROS 2 Jazzy Jalisco
*   **安装路径**: `/opt/ros/jazzy`
*   **状态**: 已安装，可直接调用。

## 4. CUDA 环境

*   **GPU 驱动**: NVIDIA-SMI 580.95.05, Driver Version 580.95.05
*   **CUDA Driver API 版本**: 13.0（宿主机驱动报告，向后兼容 Toolkit）
*   **CUDA Toolkit 版本 (nvcc)**: Release 12.9, V12.9.86
*   **编译器 (NVCC)**: 
    ```text
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2025 NVIDIA Corporation
    Built on Tue_May_27_02:21:03_PDT_2025
    Build cuda_12.9.r12.9/compiler.36037853_0
    ```
*   **CUDA 路径**: `/usr/local/cuda` (软链接)
*   **GPU 支持**: 针对 RTX 5090 (Blackwell) 优化。

## 5. Python 环境与依赖管理

本项目完全采用 `uv` 进行现代化的 Python 包管理。

*   **Python 版本**: 3.12.3
*   **包管理器**: `uv` (版本 0.9.18)
*   **配置文件**: `pyproject.toml`
*   **镜像源**: 阿里云 PyPI 镜像 (`https://mirrors.aliyun.com/pypi/simple`)

### 主要依赖 (`bitpilot`)
*   `torch >= 2.8.0`
*   `numpy >= 2.3.2`
*   `aim >= 3.29.1` (实验记录)
*   `tensorboard >= 2.20.0`
*   `matplotlib >= 3.10.6`

### CUDA 扩展 (`bitpilot._C`)
*   四旋翼动力学 CUDA 内核，编译为 `bitpilot._C` 模块。
*   **源码位置**: `src/bitpilot/_csrc/`
*   **编译配置**: 根目录 `setup.py`，使用 `torch.utils.cpp_extension.CUDAExtension`
*   **构建方式**: `uv sync` 自动编译（已配置 `no-build-isolation-package`）

### 常用命令
*   **同步环境**: `uv sync`
*   **训练**: `uv run bitpilot-train $(cat configs/single_agent.args)`
*   **推理**: `uv run bitpilot-infer --target 10 0 1.5`
*   **测试**: `uv run pytest tests/ -v`
*   **安装依赖**: `uv add <package_name>`
