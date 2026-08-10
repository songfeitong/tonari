from importlib.metadata import PackageNotFoundError, version

from .api import find_neighbors

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["find_neighbors"]
