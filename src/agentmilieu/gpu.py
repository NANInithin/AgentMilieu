"""Startup-registered ZeroGPU boundary; only serializable inference arguments cross it."""

from . import devices, llm

try:
    import spaces
except ImportError:
    spaces = None


def _compare_worker(**kwargs):
    try:
        return llm.compare_chat(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- ZeroGPU otherwise loses exception messages
        return {"error": str(exc) or type(exc).__name__}


def _probe_worker():
    try:
        return devices.probe_gpu()
    except Exception as exc:  # noqa: BLE001 -- keep worker errors serializable
        return {"error": str(exc) or type(exc).__name__}


if spaces is not None:
    compare = spaces.GPU(duration=120)(_compare_worker)
    probe = spaces.GPU(duration=15)(_probe_worker)
else:
    compare = _compare_worker
    probe = _probe_worker
