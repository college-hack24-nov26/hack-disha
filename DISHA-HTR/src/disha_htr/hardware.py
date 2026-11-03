"""Hardware detection and device selection for DISHA HTR."""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch

logger = logging.getLogger("disha_htr.hardware")


@dataclass(frozen=True)
class HardwareInfo:
    """Verified runtime hardware snapshot (no invented claims)."""

    os: str
    python_version: str
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    device_count: int
    gpu_name: str | None
    gpu_memory_total_mb: float | None
    nvidia_smi_driver: str | None
    nvidia_smi_cuda_reported: str | None
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvidia_smi_snapshot() -> tuple[str | None, str | None]:
    """Best-effort nvidia-smi probe; returns (driver, cuda_version) or Nones."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        driver = result.stdout.strip().splitlines()[0].strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, IndexError) as exc:
        logger.debug("nvidia-smi driver query failed: %s", exc)
        driver = None

    cuda_reported = None
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "CUDA Version:" in line:
                    # e.g. "... Driver Version: 581.86    CUDA Version: 13.0"
                    cuda_reported = line.split("CUDA Version:")[-1].strip().split()[0]
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("nvidia-smi full probe failed: %s", exc)

    return driver, cuda_reported


def resolve_device(preference: str = "auto", require_cuda: bool = False) -> torch.device:
    """
    Choose compute device.

    preference: auto | cuda | cpu
    Falls back to CPU when CUDA is unavailable unless require_cuda=True.
    """
    pref = (preference or "auto").lower().strip()
    cuda_ok = torch.cuda.is_available()

    if pref == "cpu":
        device = torch.device("cpu")
    elif pref == "cuda":
        if not cuda_ok:
            msg = "CUDA requested but torch.cuda.is_available() is False"
            if require_cuda:
                raise RuntimeError(msg)
            logger.warning("%s — falling back to CPU", msg)
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:  # auto
        device = torch.device("cuda" if cuda_ok else "cpu")
        if not cuda_ok:
            logger.info("CUDA not available; using CPU")

    if require_cuda and device.type != "cuda":
        raise RuntimeError("require_cuda=True but no usable CUDA device")

    return device


def detect_hardware(
    preference: str = "auto",
    require_cuda: bool = False,
) -> HardwareInfo:
    """Inspect host GPU/CUDA/Torch and return a verified HardwareInfo."""
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name: str | None = None
    gpu_memory_total_mb: float | None = None

    if cuda_available and device_count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_total_mb = round(props.total_memory / (1024 * 1024), 1)
        except Exception as exc:  # noqa: BLE001 — report honestly
            logger.warning("Could not read CUDA device properties: %s", exc)

    driver, cuda_reported = _nvidia_smi_snapshot()
    device = resolve_device(preference=preference, require_cuda=require_cuda)

    info = HardwareInfo(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_build=getattr(torch.version, "cuda", None),
        cuda_available=cuda_available,
        device_count=device_count,
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_memory_total_mb,
        nvidia_smi_driver=driver,
        nvidia_smi_cuda_reported=cuda_reported,
        selected_device=str(device),
    )
    logger.info(
        "Hardware: cuda_available=%s device=%s gpu=%s torch=%s",
        info.cuda_available,
        info.selected_device,
        info.gpu_name,
        info.torch_version,
    )
    return info
