"""Preserve the public legacy utilities while keeping benchmark imports independent."""

from importlib import import_module as _import_module
import sys as _sys
from types import ModuleType as _ModuleType


def _legacy():
    return _import_module(".legacy", __name__)


def __getattr__(name):
    if name == "__all__":
        # The former module also exported names imported by the legacy scripts.
        return tuple(key for key in vars(_legacy()) if not key.startswith("_"))
    if name == "legacy":
        return _legacy()
    if name == "benchmark" or name.startswith("__"):
        raise AttributeError(name)
    return getattr(_legacy(), name)


def __dir__():
    return sorted(set(globals()) | set(__getattr__("__all__")))


class _UtilsPackage(_ModuleType):
    """Keep assignments and patches visible to the original function globals."""

    def __setattr__(self, name, value):
        if name.startswith("__") or name in {"legacy", "benchmark"}:
            super().__setattr__(name, value)
        else:
            setattr(_legacy(), name, value)

    def __delattr__(self, name):
        if name.startswith("__") or name in {"legacy", "benchmark"}:
            super().__delattr__(name)
        else:
            delattr(_legacy(), name)


_sys.modules[__name__].__class__ = _UtilsPackage
