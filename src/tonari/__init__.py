from importlib.metadata import PackageNotFoundError, version

from .api import neighbor_list

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["neighbor_list"]
