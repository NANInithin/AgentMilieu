"""Startup-registered ZeroGPU boundary; only serializable inference arguments cross it."""

from . import llm

try:
    import spaces
except ImportError:
    spaces = None


def _compare_worker(**kwargs):
    return llm.compare_chat(**kwargs)


if spaces is not None:
    compare = spaces.GPU(duration=120)(_compare_worker)
else:
    compare = _compare_worker
