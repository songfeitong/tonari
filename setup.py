import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

extensions = [
    CppExtension(
        name="tonari._C_cpu",
        sources=[
            "csrc/bindings_cpu.cpp",
            "csrc/geometry.cpp",
            "csrc/neighbors_cpu.cpp",
        ],
        depends=["csrc/geometry.h", "csrc/neighbors_cpu.h"],
        extra_compile_args=["-O3"],
    )
]

if CUDA_HOME is not None and torch.version.cuda is not None:
    extensions.append(
        CUDAExtension(
            name="tonari._C_cuda",
            sources=[
                "csrc/bindings_cuda.cpp",
                "csrc/neighbors_cuda.cu",
                "csrc/neighbors_cell_cuda.cu",
            ],
            depends=["csrc/neighbors_cuda.h"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    )

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
