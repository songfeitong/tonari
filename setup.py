from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent


setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="torch_radius_graph._C",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "radius_graph_cuda.cu"),
                str(ROOT / "csrc" / "radius_graph_cell_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
