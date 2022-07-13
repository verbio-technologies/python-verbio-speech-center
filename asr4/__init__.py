import os

try:
    from .version import __version__  # noqa
except ImportError:
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")) as f:
        __version__ = f.read().strip()
