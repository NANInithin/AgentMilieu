import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmilieu import devices
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


def test_cpu_probe_meminfo(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, **_: MEMINFO)
    profile = devices.probe_cpu()
    assert profile.name == "This machine CPU RAM (probed)"
    assert profile.total_ram == 6000000 * 1024
    assert profile.available_ram == 4000000 * 1024


def test_cpu_probe_psutil_fallback(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, **_: "invalid")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=8000, available=6000)))
    assert devices.probe_cpu() == DeviceProfile(8000, 6000, "This machine CPU RAM (probed)")
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(RuntimeError, match="Cannot probe CPU RAM"):
        devices.probe_cpu()


def test_gpu_probe_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, mem_get_info=lambda: (30 * devices.GIB, 48 * devices.GIB),
        get_device_name=lambda: "GPU example")))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("torch was sufficient"))
    assert devices.probe_gpu() == {"name": "GPU example", "total_ram": 48 * devices.GIB,
                                   "available_ram": 30 * devices.GIB}


def test_gpu_probe_nvidia_cli(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    def run(args, **kwargs):
        assert args == ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout='"GPU, example", 49152, 32768\nsecond GPU, 1024, 512\n')

    monkeypatch.setattr(subprocess, "run", run)
    assert devices.probe_gpu() == {"name": "GPU, example", "total_ram": 48 * devices.GIB,
                                   "available_ram": 32 * devices.GIB}


@pytest.mark.parametrize("output", ["", "bad,row", "name, 1024, 2048", "name, bad, 10"])
def test_gpu_probe_invalid_output_is_data(monkeypatch, output):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=output))
    assert "error" in devices.probe_gpu()


def test_gpu_probe_missing_command_and_cpu_budget(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    def fail(*_args, **_kwargs):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(Path, "read_text", lambda self, **_: MEMINFO)
    info = devices.probe_gpu()
    assert "error" in info
    profile, summary = devices.probe_machine(info)
    assert profile.name == "This machine CPU RAM (probed)"
    assert "CPU RAM:" in summary and "Budget uses: This machine CPU RAM (probed)" in summary


def test_gpu_profile_preferred_over_cpu(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, **_: MEMINFO)
    info = {"name": "GPU example", "total_ram": 48 * devices.GIB, "available_ram": 30 * devices.GIB}
    profile, summary = devices.probe_machine(info)
    assert profile == DeviceProfile(48 * devices.GIB, 30 * devices.GIB, "GPU example (probed VRAM)")
    assert "CPU RAM:" in summary and "GPU:" in summary and "Budget uses: GPU example" in summary

    def fail():
        raise RuntimeError("RAM unavailable")

    monkeypatch.setattr(devices, "probe_cpu", fail)
    assert devices.probe_machine(info)[0] == profile
    with pytest.raises(RuntimeError, match="Choose Preset or Manual"):
        devices.probe_machine({"error": "GPU unavailable"})
