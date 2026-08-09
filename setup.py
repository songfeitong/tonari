from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="torch_radius_graph._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/radius_graph_cuda.cu",
                "csrc/radius_graph_cell_cuda.cu",
            ],
            depends=["csrc/radius_graph_cuda.h"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
