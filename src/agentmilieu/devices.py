"""RAM profiles use bytes throughout; phone presets are illustrative, not live probes."""

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class DeviceProfile:
    total_ram: int
    available_ram: int
    name: str = "manual"

    def __post_init__(self):
        if not 0 <= self.available_ram <= self.total_ram or self.total_ram <= 0:
            raise ValueError("RAM must satisfy 0 <= available_ram <= total_ram and total_ram > 0")

    @classmethod
    def from_meminfo(cls, text: str, name: str = "adb device"):
        fields = dict(re.findall(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB\s*$", text, re.MULTILINE))
        if not {"MemTotal", "MemAvailable"} <= fields.keys():
            raise ValueError("meminfo requires MemTotal and MemAvailable in kB")
        return cls(int(fields["MemTotal"]) * 1024, int(fields["MemAvailable"]) * 1024, name)


def probe_adb(*, timeout: float = 5) -> DeviceProfile | None:
    """Return None for missing adb, disconnected/unauthorized devices, or timeouts.

    A successful command with malformed meminfo raises ValueError.
    """
    try:
        result = subprocess.run(["adb", "shell", "cat", "/proc/meminfo"],
                                capture_output=True, text=True, check=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return DeviceProfile.from_meminfo(result.stdout)


def probe_cpu() -> DeviceProfile:
    name = "This machine CPU RAM (probed)"
    try:
        return DeviceProfile.from_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"), name)
    except (OSError, ValueError):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Cannot probe CPU RAM: /proc/meminfo unavailable; install psutil") from exc
        memory = psutil.virtual_memory()
        return DeviceProfile(int(memory.total), int(memory.available), name)


def probe_gpu() -> dict:
    """Call inside the GPU worker on ZeroGPU; return only serializable data."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            profile = DeviceProfile(int(total), int(free), torch.cuda.get_device_name())
            return {"name": profile.name, "total_ram": profile.total_ram,
                    "available_ram": profile.available_ram}
    except Exception:  # noqa: BLE001, S110 -- torch/driver failures must allow the CLI fallback
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5)
        row = next(csv.reader(result.stdout.splitlines()))
        name, total, free = row
        profile = DeviceProfile(int(float(total) * 1024**2), int(float(free) * 1024**2), name.strip())
        return {"name": profile.name, "total_ram": profile.total_ram,
                "available_ram": profile.available_ram}
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as exc:
        return {"error": f"No GPU available: {exc}"}


def probe_machine(gpu_info) -> tuple[DeviceProfile, str]:
    rows = []
    cpu = None
    try:
        cpu = probe_cpu()
        rows.append(f"CPU RAM: {cpu.total_ram / GIB:.2f} GiB total, {cpu.available_ram / GIB:.2f} GiB available")
    except (RuntimeError, OSError, ValueError) as exc:
        rows.append(f"CPU RAM probe unavailable: {exc}")
    if "error" not in gpu_info:
        device = DeviceProfile(gpu_info["total_ram"], gpu_info["available_ram"],
                               gpu_info["name"] + " (probed VRAM)")
        rows.insert(0, f"GPU: {device.name}; {device.total_ram / GIB:.2f} GiB total, "
                       f"{device.available_ram / GIB:.2f} GiB free")
    elif cpu is not None:
        device = cpu
        rows.insert(0, gpu_info["error"])
    else:
        raise RuntimeError("Cannot probe a device. Choose Preset or Manual. " + "\n".join(rows))
    rows.append(f"Budget uses: {device.name}")
    return device, "\n".join(rows)


DEVICE_PRESETS = {
    f"phone-{gb}gb": DeviceProfile(gb * GIB, int(gb * GIB * 0.75), f"{gb} GB phone (estimate)")
    for gb in (4, 6, 8, 12)
}
