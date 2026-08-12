from __future__ import annotations

import pytest
import torch

from tonari import _extensions


def test_cpu_build_rejects_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_extensions, "BUILD_WITH_CUDA", False)
    with pytest.raises(ImportError, match="BUILD_CUDA=1"):
        _extensions._require_cuda_provider()


def test_torch_major_minor_mismatch_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_extensions, "BUILD_TORCH_VERSION", "1.13.0")
    with pytest.raises(ImportError, match="built against PyTorch 1.13.0"):
        _extensions._check_torch_compatibility(cuda=False)


def test_torch_patch_version_is_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    major, minor = torch.__version__.split(".", maxsplit=2)[:2]
    monkeypatch.setattr(
        _extensions, "BUILD_TORCH_VERSION", f"{major}.{minor}.0+different"
    )
    _extensions._check_torch_compatibility(cuda=False)


def test_torch_cuda_runtime_mismatch_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_extensions, "BUILD_TORCH_VERSION", torch.__version__)
    monkeypatch.setattr(_extensions, "BUILD_TORCH_CUDA_VERSION", "0.0")
    with pytest.raises(ImportError, match="PyTorch CUDA runtime 0.0"):
        _extensions._check_torch_compatibility(cuda=True)
