# Source builds

Tonari uses CMake as the only native build description and scikit-build-core as its PEP 517 backend. Every Python installation builds the NumPy CPU and Torch CPU providers from the reusable CMake core targets. CUDA is enabled by default and can be disabled explicitly with `BUILD_CUDA=0`.

| Build | Native providers | Requirements |
| --- | --- | --- |
| CUDA, default | NumPy CPU, Torch CPU, Torch CUDA | C++20 compiler, CUDA-enabled PyTorch, CUDA toolkit containing `nvcc` |
| CPU | NumPy CPU, Torch CPU | C++20 compiler, PyTorch |

## Regular and editable installs

```bash
# Default CUDA build
python -m pip install .
python -m pip install --editable .

# Explicit CPU-only build
BUILD_CUDA=0 python -m pip install .
BUILD_CUDA=0 python -m pip install --editable .
```

`BUILD_CUDA` accepts only `0` or `1`. The default build fails with an actionable error if the build environment contains CPU-only PyTorch or no CUDA compiler; it never silently produces a CPU build. `BUILD_CUDA=0` controls Tonari's provider set and does not select the PyTorch distribution. Reinstall after changing this option, the PyTorch major/minor version, or the PyTorch CUDA runtime.

Editable installations keep Python sources in `src/tonari` and compiled/generated files in the installation managed by scikit-build-core; native artifacts are not written into the source package. Consequently a CPU reinstall does not need a source-tree cleanup shim.

## Wheel builds

```bash
BUILD_CUDA=0 python -m build --wheel
python -m build --wheel
```

The wheel is assembled from CMake install rules. It contains `_numpy_cpu`, `_torch_cpu`, optional `_torch_cuda`, and a generated `tonari._build_info` module alongside the Python sources. The metadata records the exact PyTorch package version, PyTorch CUDA runtime, local CUDA toolkit version, and whether the CUDA provider was built. Provider loading accepts PyTorch patch updates within the same major/minor line and requires the CUDA runtime to match for the CUDA provider.

## CUDA architectures

CMake reads PyTorch's native include/library layout and libstdc++ ABI directly from the selected build-environment package. CUDA code generation defaults to the visible build device. Set CMake's standard `CUDAARCHS` environment variable when producing a wheel for other machines or a deliberate deployment baseline, for example:

```bash
CUDAARCHS="80-real;90-real;120" python -m build --wheel
```

Numeric entries without a suffix emit both real code and PTX; `-real` and `-virtual` select one form explicitly. Each extra architecture increases build time and wheel size. The supported values are constrained by the local toolkit. Tonari requires the toolkit and PyTorch CUDA runtime to use the same major version; a minor toolkit update is accepted.

## Reusing the native core

When Tonari is added below another CMake project, Python providers are disabled by default and the framework-neutral targets remain available:

```cmake
add_subdirectory(vendor/tonari)
target_link_libraries(elfes_native PRIVATE tonari::cpu)
```

`tonari::geometry` provides periodic metadata only. `tonari::cpu` adds CPU neighbor search and the reusable worker pool. Neither target installs Python modules or assumes a wheel layout. A parent that deliberately wants the provider modules may set `TONARI_BUILD_PYTHON=ON` before `add_subdirectory`; it may then control CUDA with the CMake option `TONARI_BUILD_CUDA`.

For a direct core-only development build:

```bash
cmake -S . -B build/core -G Ninja -DTONARI_BUILD_PYTHON=OFF
cmake --build build/core
```

## CI

Hosted CI sets `BUILD_CUDA=0`, builds the reusable core directly, installs the standalone package through scikit-build-core, and tests NumPy and Torch CPU behavior across supported Python versions. The CUDA job uses the default CUDA build on a self-hosted runner when the repository variable `TONARI_CUDA_RUNNER` is `true`.
