"""
setuptools 构建脚本 — 仅用于编译 CUDA C++ 扩展 bitpilot._C。
项目元数据、依赖、入口点等全部在 pyproject.toml 中配置。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            "bitpilot._C",
            [
                "src/bitpilot/_csrc/quadsim.cpp",
                "src/bitpilot/_csrc/quadsim_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
