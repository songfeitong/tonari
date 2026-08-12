from __future__ import annotations

import re
from functools import lru_cache
from importlib import import_module
from types import ModuleType

BUILD_WITH_CUDA: bool | None
BUILD_TORCH_VERSION: str | None
BUILD_TORCH_CUDA_VERSION: str | None
BUILD_CUDA_TOOLKIT_VERSION: str | None

try:
    from . import _build_info
except ImportError:
    BUILD_WITH_CUDA = None
    BUILD_TORCH_VERSION = None
    BUILD_TORCH_CUDA_VERSION = None
    BUILD_CUDA_TOOLKIT_VERSION = None
else:
    BUILD_WITH_CUDA = _build_info.BUILD_WITH_CUDA
    BUILD_TORCH_VERSION = _build_info.BUILD_TORCH_VERSION
    BUILD_TORCH_CUDA_VERSION = _build_info.BUILD_TORCH_CUDA_VERSION
    BUILD_CUDA_TOOLKIT_VERSION = _build_info.BUILD_CUDA_TOOLKIT_VERSION


def _load(name: str) -> ModuleType:
    return import_module(f".{name}", package=__package__)


def _major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ImportError(f"could not parse PyTorch version {version!r}")
    return int(match.group(1)), int(match.group(2))


def _require_cuda_provider() -> None:
    if BUILD_WITH_CUDA is False:
        raise ImportError(
            "the CUDA provider was not built; reinstall tonari with BUILD_CUDA=1"
        )


def _check_torch_compatibility(*, cuda: bool) -> None:
    import torch

    if BUILD_TORCH_VERSION is not None and _major_minor(
        torch.__version__
    ) != _major_minor(BUILD_TORCH_VERSION):
        raise ImportError(
            "the tonari Torch provider was built against PyTorch "
            f"{BUILD_TORCH_VERSION}, but PyTorch {torch.__version__} is installed; "
            "reinstall tonari against the current PyTorch environment"
        )
    if cuda and BUILD_TORCH_CUDA_VERSION != torch.version.cuda:
        raise ImportError(
            "the tonari CUDA provider was built for the PyTorch CUDA runtime "
            f"{BUILD_TORCH_CUDA_VERSION}, but the installed PyTorch uses "
            f"{torch.version.cuda}; reinstall tonari against the current PyTorch "
            "environment"
        )


@lru_cache
def load_numpy_cpu() -> ModuleType:
    return _load("_numpy_cpu")


@lru_cache
def load_torch_cpu() -> ModuleType:
    _check_torch_compatibility(cuda=False)
    return _load("_torch_cpu")


@lru_cache
def load_torch_cuda() -> ModuleType:
    _require_cuda_provider()
    _check_torch_compatibility(cuda=True)
    return _load("_torch_cuda")
