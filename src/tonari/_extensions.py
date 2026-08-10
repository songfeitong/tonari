from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType


def _load(name: str) -> ModuleType:
    return import_module(f".{name}", package=__package__)


@lru_cache
def load_numpy_cpu() -> ModuleType:
    return _load("_numpy_cpu")


@lru_cache
def load_torch_cpu() -> ModuleType:
    return _load("_torch_cpu")


@lru_cache
def load_torch_cuda() -> ModuleType:
    return _load("_torch_cuda")
