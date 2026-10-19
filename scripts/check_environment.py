"""Report the resources and tools available for developing and running GroundedAI.

Prints facts only and never prints secret values: environment variables and proxy
settings are reported as "set"/"not set". Nothing is downloaded or installed.

    python scripts/check_environment.py          # human-readable
    python scripts/check_environment.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_MODELS = [
    ("BAAI/bge-small-en-v1.5", "embedding (default dense model, ~130 MB)"),
    ("sentence-transformers/all-MiniLM-L6-v2", "embedding (alternative, ~90 MB)"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker (~90 MB)"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "local transformers LLM (~1 GB)"),
]


def memory_gb() -> tuple[float | None, float | None]:
    """(total, available) RAM in GB."""
    if platform.system() == "Windows":

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
        if getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # noqa: B009
            return round(status.ullTotalPhys / 1024**3, 1), round(status.ullAvailPhys / 1024**3, 1)
        return None, None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) / 1024**2
        return round(values.get("MemTotal", 0), 1), round(values.get("MemAvailable", 0), 1)
    return None, None


def tool_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if not executable:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed, trusted commands
            [executable, *command[1:]], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "installed (version check failed)"
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else "installed"


def docker_daemon() -> bool | None:
    executable = shutil.which("docker")
    if not executable:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def ollama_server(url: str) -> dict[str, Any]:
    # Bypass HTTP proxies for this local check (a proxy cannot reach localhost).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read())
        return {"reachable": True, "models": [m.get("name") for m in payload.get("models", [])]}
    except (urllib.error.URLError, OSError, ValueError):
        return {"reachable": False, "models": []}


def hf_cache_models() -> dict[str, bool]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return {}
    return {
        model: isinstance(try_to_load_from_cache(model, "config.json"), str)
        for model, _ in KNOWN_MODELS
    }


def gpu() -> str:
    if shutil.which("nvidia-smi"):
        version = tool_version(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        return f"NVIDIA: {version}"
    return "no CUDA GPU detected (CPU-only operation)"


def collect() -> dict[str, Any]:
    total, available = memory_gb()
    disk = shutil.disk_usage(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    try:
        from groundedai.evaluation.environment import cpu_name, power_source

        cpu, power = cpu_name(), power_source()
    except ImportError:
        cpu, power = platform.processor() or "unknown", "unknown"
    base_url = os.environ.get("GROUNDEDAI_OLLAMA_BASE_URL", "http://localhost:11434")
    return {
        "workspace": {
            "path": str(REPO_ROOT),
            "git_repository": (REPO_ROOT / ".git").exists(),
            "disk_free_gb": round(disk.free / 1024**3, 1),
            "disk_total_gb": round(disk.total / 1024**3, 1),
        },
        "machine": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "ram_total_gb": total,
            "ram_available_gb": available,
            "gpu": gpu(),
            "power_source": power,
        },
        "python": {
            "version": platform.python_version(),
            "executable_in_venv": sys.prefix != sys.base_prefix,
            "groundedai_installed": importlib.util.find_spec("groundedai") is not None,
            "ml_extra_installed": all(
                importlib.util.find_spec(name) is not None
                for name in ("sentence_transformers", "transformers", "langchain_huggingface")
            ),
            "pgserver_installed": importlib.util.find_spec("pgserver") is not None,
        },
        "tools": {
            "git": tool_version(["git", "--version"]),
            "node": tool_version(["node", "--version"]),
            "npm": tool_version(["npm", "--version"]),
            "docker_cli": tool_version(["docker", "--version"]),
            "docker_daemon_running": docker_daemon(),
            "ollama_cli": tool_version(["ollama", "--version"]),
            "ollama_server": {"url": base_url, **ollama_server(base_url)},
        },
        "huggingface_cache": hf_cache_models(),
        "configuration": {
            "groundedai_variables_set": sorted(
                k for k in os.environ if k.startswith("GROUNDEDAI_")
            ),
            "http_proxy_set": bool(os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")),
            "https_proxy_set": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")),
            "no_proxy_set": bool(os.environ.get("NO_PROXY") or os.environ.get("no_proxy")),
            "dotenv_file_present": (REPO_ROOT / ".env").is_file(),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    for section, values in report.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            print(f"  {key:28s} {value}")
    print("\nNotes:")
    machine, tools = report["machine"], report["tools"]
    if machine["ram_total_gb"] and machine["ram_total_gb"] < 12:
        print("  - Less than 12 GB RAM: prefer the hashing embedder and qwen2.5:0.5b with Ollama.")
    if not tools["docker_daemon_running"]:
        print("  - Docker daemon not running: use the SQLite setup (no Docker needed).")
    if not tools["ollama_server"]["reachable"]:
        print("  - No Ollama server: the app runs with the extractive demo generator (not an LLM).")
    if report["configuration"]["http_proxy_set"] and not report["configuration"]["no_proxy_set"]:
        print(
            "  - HTTP proxy set without NO_PROXY: "
            "GroundedAI already bypasses it for localhost URLs."
        )
    if machine["power_source"] == "battery":
        print("  - Running on battery: latency measurements will not be comparable.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()
    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
