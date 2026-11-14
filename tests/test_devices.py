import subprocess
from types import SimpleNamespace

import pytest

from agentmilieu.devices import DEVICE_PRESETS, DeviceProfile, probe_adb

MEMINFO = "MemTotal:       6000000 kB\nMemFree: 10 kB\nMemAvailable: 4000000 kB\n"


def test_meminfo():
    device = DeviceProfile.from_meminfo(MEMINFO)
    assert device.total_ram == 6000000 * 1024
    assert device.available_ram == 4000000 * 1024
    assert len(DEVICE_PRESETS) == 4


@pytest.mark.parametrize("text", ["", "MemTotal: 2 kB", "MemTotal: 2 MB\nMemAvailable: 1 MB",
                                  "MemTotal: 1 kB\nMemAvailable: 2 kB"])
def test_bad_meminfo(text):
    with pytest.raises(ValueError):
        DeviceProfile.from_meminfo(text)


def test_probe(monkeypatch):
    def run(args, **kwargs):
        assert args == ["adb", "shell", "cat", "/proc/meminfo"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout=MEMINFO)
    monkeypatch.setattr(subprocess, "run", run)
    assert probe_adb().available_ram == 4000000 * 1024


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(1, "adb"),
                                  subprocess.TimeoutExpired("adb", 5)])
def test_probe_unavailable(monkeypatch, error):
    def run(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, "run", run)
    assert probe_adb() is None
