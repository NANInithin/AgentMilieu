"""RAM profiles use bytes throughout; phone presets are illustrative, not live probes."""

import re
import subprocess
from dataclasses import dataclass

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


DEVICE_PRESETS = {
    f"phone-{gb}gb": DeviceProfile(gb * GIB, int(gb * GIB * 0.75), f"{gb} GB phone (estimate)")
    for gb in (4, 6, 8, 12)
}
