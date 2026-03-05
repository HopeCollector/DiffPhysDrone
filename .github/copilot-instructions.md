# 项目环境与配置说明

这份文档记录了本项目的运行环境、依赖配置以及系统探测结果。

## 1. 项目概览

*   **名称**: Vision-based Agile Flight Training Code (DiffPhy)
*   **描述**: 基于可微物理（Differentiable Physics）的视觉敏捷飞行训练代码。
*   **硬件适配**: 已针对 **NVIDIA RTX 5090** 进行适配。
*   **核心模块**:
    *   `diffphy`: 主训练框架。
    *   `quadsim_cuda`: 自定义的 CUDA 加速四旋翼动力学仿真模块（C++/CUDA 扩展）。

## 2. 系统环境 (System Environment)

*   **操作系统**: Ubuntu 24.04.2 LTS (Noble Numbat)
*   **运行环境**: Docker 容器环境
*   **内核信息**: Linux (运行在容器内)

## 3. ROS2 环境

*   **版本 Distribution**: ROS 2 Jazzy Jalisco
*   **安装路径**: `/opt/ros/jazzy`
*   **状态**: 已安装，可直接调用。

## 4. CUDA 环境

*   **CUDA Toolkit 版本**: Release 12.9, V12.9.86
*   **编译器 (NVCC)**: 
    ```text
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2025 NVIDIA Corporation
    Built on Tue_May_27_02:21:03_PDT_2025
    Build cuda_12.9.r12.9/compiler.36037853_0
    ```
*   **CUDA 路径**: `/usr/local/cuda-12.9` (软链接至 `/usr/local/cuda`)
*   **GPU 支持**: 针对 RTX 5090 优化。

## 5. Python 环境与依赖管理

本项目完全采用 `uv` 进行现代化的 Python 包管理。

*   **Python 版本**: 3.12.3
*   **包管理器**: `uv` (版本 0.9.18)
*   **配置文件**: `pyproject.toml`
*   **镜像源**: 阿里云 PyPI 镜像 (`https://mirrors.aliyun.com/pypi/simple`)

### 主要依赖 (`diffphy`)
*   `torch >= 2.8.0`
*   `numpy >= 2.3.2`
*   `aim >= 3.29.1` (实验记录)
*   `tensorboard >= 2.20.0`
*   `matplotlib >= 3.10.6`

### 本地扩展 (`quadsim-cuda`)
*   这是一个包含 CUDA内核的 Python 扩展。
*   **源码位置**: `./quadsim_cuda`
*   **编译后端**: `setuptools`, `wheel`
*   **构建依赖**: `torch`, `numpy`

### 常用命令
*   **同步环境**: `uv sync`
*   **运行代码**: `uv run main_cuda.py`
*   **安装依赖**: `uv add <package_name>`
