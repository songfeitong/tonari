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
        name="torch_radius_graph._C_cpu",
        sources=[
            "csrc/bindings_cpu.cpp",
            "csrc/periodic_geometry.cpp",
            "csrc/radius_graph_cpu.cpp",
        ],
        depends=["csrc/periodic_geometry.h", "csrc/radius_graph_cpu.h"],
        extra_compile_args=["-O3"],
    )
]

if CUDA_HOME is not None and torch.version.cuda is not None:
    extensions.append(
        CUDAExtension(
            name="torch_radius_graph._C_cuda",
            sources=[
                "csrc/bindings_cuda.cpp",
                "csrc/radius_graph_cuda.cu",
                "csrc/radius_graph_cell_cuda.cu",
            ],
            depends=["csrc/radius_graph_cuda.h"],
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
