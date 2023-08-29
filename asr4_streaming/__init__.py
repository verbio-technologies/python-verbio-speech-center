import os

try:
    import importlib.metadata as importlib_metadata  # noqa

    __version__ = importlib_metadata.version(__package__ or __name__)
except ImportError:
    try:
        import importlib_metadata  # noqa

        __version__ = importlib_metadata.version(__package__ or __name__)
    except ImportError:
        with open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")
        ) as f:
            __version__ = f.read().strip()
