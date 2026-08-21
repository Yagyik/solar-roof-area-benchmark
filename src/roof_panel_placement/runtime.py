"""Small runtime checks and reproducibility snapshots for Colab runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "Pillow",
    "matplotlib",
    "PyYAML",
    "torch",
    "torchvision",
    "transformers",
    "huggingface-hub",
)


def require_runtime(
    accelerator: str,
    python_major_minor: str | list[str],
    minimum_gpu_memory_gb: float | None = None,
) -> dict:
    """Fail early when the active Colab runtime cannot run an experiment."""
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    allowed_python = (
        [python_major_minor] if isinstance(python_major_minor, str) else python_major_minor
    )
    if actual_python not in allowed_python:
        raise RuntimeError(
            f"This run requires Python in {allowed_python}; found {actual_python}."
        )

    accelerator = accelerator.lower()
    cuda_available = False
    gpu_name = None
    gpu_memory_gb = None
    if accelerator == "gpu":
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except ImportError:
            cuda_available = False
        if not cuda_available:
            raise RuntimeError(
                "This experiment requires a GPU. In VS Code select a Colab GPU "
                "runtime before continuing."
            )
        if minimum_gpu_memory_gb is not None and gpu_memory_gb < minimum_gpu_memory_gb:
            raise RuntimeError(
                f"This experiment requires at least {minimum_gpu_memory_gb:.1f} GiB GPU "
                f"memory; found {gpu_memory_gb:.1f} GiB on {gpu_name}."
            )
    elif accelerator != "cpu":
        raise ValueError(f"Unsupported accelerator requirement: {accelerator}")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "accelerator_required": accelerator,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
    }


def git_revision(repository_root: Path) -> str | None:
    """Return the checked-out Git revision when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_snapshot(repository_root: Path) -> dict:
    """Capture the compact environment record saved with every run."""
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "git_revision": git_revision(repository_root),
        "packages": versions,
    }


def write_json(payload: dict, path: Path) -> None:
    def portable_scalar(value: object) -> object:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=portable_scalar),
        encoding="utf-8",
    )
