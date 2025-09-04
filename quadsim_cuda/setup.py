from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            'quadsim_cuda.quadsim_cuda',
            [
                'src/quadsim_cuda/cc/quadsim.cpp',
                'src/quadsim_cuda/cc/quadsim_kernel.cu',
                'src/quadsim_cuda/cc/dynamics_kernel.cu',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)