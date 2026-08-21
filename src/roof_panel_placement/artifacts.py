"""Persistent, downloadable experiment bundles stored in Google Drive."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunDirectory:
    path: Path
    run_id: str

    @property
    def figures(self) -> Path:
        return self.path / "representative_examples"


def create_run_directory(drive_root: Path, artifact_subdirectory: str, run_name: str) -> RunDirectory:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name}_{timestamp}"
    path = drive_root / artifact_subdirectory / run_id
    (path / "representative_examples").mkdir(parents=True, exist_ok=False)
    (path / "sensitivity").mkdir()
    return RunDirectory(path=path, run_id=run_id)


def zip_run_directory(run: RunDirectory) -> Path:
    """Create a sibling ZIP that remains downloadable after Colab disconnects."""
    archive = shutil.make_archive(str(run.path), "zip", root_dir=run.path.parent, base_dir=run.path.name)
    return Path(archive)


def save_probability_maps(cases: list[dict], directory: Path) -> Path:
    """Persist compact per-image probability maps for retrospective analysis."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        sample_id = str(case["sample_id"])
        probability = np.asarray(case["probability"], dtype=np.float16)
        output_path = directory / f"{sample_id}.npz"
        np.savez_compressed(output_path, probability=probability)
        records.append(
            {
                "sample_id": sample_id,
                "file": output_path.name,
                "dtype": str(probability.dtype),
                "shape": list(probability.shape),
            }
        )

    index_path = directory / "index.json"
    index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return index_path


def save_binary_masks(cases: list[dict], directory: Path, mask_keys: tuple[str, ...]) -> Path:
    """Persist named binary masks for later visual and numerical examination."""
    if not mask_keys:
        raise ValueError("At least one mask key is required.")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        sample_id = str(case["sample_id"])
        payload = {
            key: np.asarray(case[key], dtype=np.uint8)
            for key in mask_keys
            if key in case
        }
        if not payload:
            raise ValueError(f"Case {sample_id} contains none of the requested masks.")
        output_path = directory / f"{sample_id}.npz"
        np.savez_compressed(output_path, **payload)
        records.append(
            {
                "sample_id": sample_id,
                "file": output_path.name,
                "masks": {
                    key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                    for key, value in payload.items()
                },
            }
        )

    index_path = directory / "index.json"
    index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return index_path


def write_artifact_manifest(run: RunDirectory) -> Path:
    """Write file sizes and SHA-256 hashes before the run directory is zipped."""
    output_path = run.path / "artifact_manifest.json"
    records = []
    for path in sorted(run.path.rglob("*")):
        if not path.is_file() or path == output_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": str(path.relative_to(run.path)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )

    payload = {"run_id": run.run_id, "files": records}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
