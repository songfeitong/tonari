# Source builds

Tonari is a Torch package with NumPy interoperability. Every installation builds the NumPy CPU and Torch CPU providers. CUDA is enabled by default and can be disabled explicitly with `BUILD_CUDA=0`.

| Build | Native providers | Requirements |
| --- | --- | --- |
| CUDA, default | NumPy CPU, Torch CPU, Torch CUDA | C++20 compiler, CUDA-enabled PyTorch, CUDA toolkit containing `nvcc` |
| CPU | NumPy CPU, Torch CPU | C++20 compiler, PyTorch |

## Regular installs

```bash
python -m pip install .
BUILD_CUDA=0 python -m pip install .
```

The default CUDA build fails with an actionable error if PyTorch has no CUDA runtime or the local CUDA toolkit is unavailable. It never silently produces a CPU build. `BUILD_CUDA` accepts only `0` or `1`.

Use `--force-reinstall --no-cache-dir` when changing the CUDA option in an existing environment so that pip does not reuse a wheel produced with the previous setting.

## Editable installs

```bash
python -m pip install --editable .
BUILD_CUDA=0 python -m pip install --editable .
```

Changing an editable installation to a CPU build removes a stale Tonari CUDA extension from the source tree.

## PyTorch CPU installations

`BUILD_CUDA=0` controls whether Tonari compiles its CUDA extension; it does not select which PyTorch distribution is installed. A user who needs a CPU-only PyTorch installation must select the appropriate PyTorch package index before installing Tonari.

## Build metadata and compatibility

Every compiled installation contains a generated `tonari._build_info` module recording whether CUDA was built, the PyTorch version, the PyTorch CUDA runtime, and the local CUDA toolkit version. Provider loading checks this metadata before importing a Torch extension.

Torch patch releases within the same major/minor line are accepted. A different Torch major/minor version requires rebuilding Tonari. The CUDA provider additionally requires the current PyTorch CUDA runtime to match the runtime used for the build.

## CI

Hosted CI builds with `BUILD_CUDA=0` and tests NumPy and Torch CPU behavior across all supported Python versions. The CUDA job uses a self-hosted CUDA runner when the repository variable `TONARI_CUDA_RUNNER` is set to `true`.
