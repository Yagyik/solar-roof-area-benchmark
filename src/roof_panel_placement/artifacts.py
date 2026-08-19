"""Persistent, downloadable experiment bundles stored in Google Drive."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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

