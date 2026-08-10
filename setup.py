import torch
from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

extensions = [
    Pybind11Extension(
        name="tonari._numpy_cpu",
        sources=[
            "csrc/numpy/bindings.cpp",
            "csrc/core/geometry.cpp",
            "csrc/core/neighbors_cpu.cpp",
        ],
        depends=[
            "csrc/core/errors.h",
            "csrc/core/geometry.h",
            "csrc/core/neighbors_cpu.h",
            "csrc/core/pair_policy.h",
        ],
        cxx_std=20,
        extra_compile_args=["-O3"],
    ),
    CppExtension(
        name="tonari._torch_cpu",
        sources=[
            "csrc/torch/bindings_cpu.cpp",
            "csrc/core/geometry.cpp",
            "csrc/core/neighbors_cpu.cpp",
        ],
        depends=[
            "csrc/core/errors.h",
            "csrc/core/geometry.h",
            "csrc/core/neighbors_cpu.h",
            "csrc/core/pair_policy.h",
        ],
        extra_compile_args=["-O3"],
    ),
]

if CUDA_HOME is not None and torch.version.cuda is not None:
    extensions.append(
        CUDAExtension(
            name="tonari._torch_cuda",
            sources=[
                "csrc/torch/bindings_cuda.cpp",
                "csrc/core/geometry.cpp",
                "csrc/torch/neighbors_cuda.cu",
                "csrc/torch/neighbors_cell_cuda.cu",
            ],
            depends=[
                "csrc/core/errors.h",
                "csrc/core/geometry.h",
                "csrc/core/pair_policy.h",
                "csrc/torch/neighbors_cuda.h",
            ],
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
