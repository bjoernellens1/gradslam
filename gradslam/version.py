"""Package version helper."""

from importlib import metadata

try:
    __version__ = metadata.version("opengradslam")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+editable"
