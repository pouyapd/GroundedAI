"""Best-effort description of the machine that produced a measurement.

Recorded in every benchmark report so latency numbers can be interpreted.
No identifiers (host name, user name, paths) are collected.
"""

from __future__ import annotations

import ctypes
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _kernel32() -> Any:
    return getattr(ctypes, "windll").kernel32  # noqa: B009 - only exists on Windows


PACKAGES = (
    "groundedai",
    "langchain-core",
    "langchain-text-splitters",
    "langchain-ollama",
    "langchain-huggingface",
    "sentence-transformers",
    "transformers",
    "torch",
    "numpy",
    "sqlalchemy",
    "fastapi",
)


def cpu_name() -> str:
    if _is_windows():
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            return platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def total_ram_gb() -> float | None:
    if _is_windows():

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(float(status.ullTotalPhys) / 1024**3, 1)
        return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    return None


def power_source() -> str:
    """'ac', 'battery' or 'unknown'. Laptops throttle on battery, so latency depends on it."""
    if _is_windows():

        class PowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = PowerStatus()
        if _kernel32().GetSystemPowerStatus(ctypes.byref(status)):
            return {1: "ac", 0: "battery"}.get(status.ACLineStatus, "unknown")
        return "unknown"
    supplies = Path("/sys/class/power_supply")
    if supplies.is_dir():
        for supply in supplies.iterdir():
            online = supply / "online"
            kind = supply / "type"
            if kind.is_file() and kind.read_text().strip() == "Mains" and online.is_file():
                return "ac" if online.read_text().strip() == "1" else "battery"
    return "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def environment_info() -> dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "gpu_used": False,
        "power_source": power_source(),
        "packages": package_versions(),
    }


def hf_model_revision(model_id: str) -> str | None:
    """Commit hash of a model in the local Hugging Face cache, if present."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return None
