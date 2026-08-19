"""RID2 paths and image/mask loading shared by all segmentation methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Rid2Paths:
    root: Path

    @property
    def image_dir(self) -> Path:
        return self.root / "images"

    @property
    def roof_mask_dir(self) -> Path:
        return self.root / "masks" / "masks_segments"

    @property
    def training_split(self) -> Path:
        return self.root / "training_split_512.csv"

    def image_path(self, image_name: str) -> Path:
        return self.image_dir / image_name

    def roof_mask_path(self, image_name: str) -> Path:
        return self.roof_mask_dir / image_name


def is_rid2_root(path: Path) -> bool:
    required = (
        path / "images",
        path / "masks" / "masks_segments",
        path / "training_split_512.csv",
    )
    return all(item.exists() for item in required)


def find_rid2_archive(drive_root: Path) -> Path:
    """Locate the persistent archive without embedding a user-specific variant."""
    candidates = (
        drive_root / "roof_information_dataset_2.zip",
        drive_root / "data" / "archives" / "roof_information_dataset_2.zip",
        drive_root / "data_downloads" / "roof_information_dataset_2.zip",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(drive_root.rglob("roof_information_dataset_2.zip"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find roof_information_dataset_2.zip under {drive_root}."
        )
    return matches[0]


def prepare_rid2(drive_root: Path, work_root: Path) -> Rid2Paths:
    """Extract RID2 to fast ephemeral storage once per fresh Colab runtime."""
    if is_rid2_root(work_root):
        return Rid2Paths(work_root)

    work_root.mkdir(parents=True, exist_ok=True)
    archive_path = find_rid2_archive(drive_root)
    with ZipFile(archive_path) as archive:
        archive.extractall(work_root)

    if is_rid2_root(work_root):
        return Rid2Paths(work_root)

    split_files = list(work_root.rglob("training_split_512.csv"))
    valid_roots = [path.parent for path in split_files if is_rid2_root(path.parent)]
    if not valid_roots:
        raise RuntimeError(f"RID2 archive did not produce a valid dataset under {work_root}.")
    return Rid2Paths(valid_roots[0])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_roof_mask(path: Path, background_value: int = 5) -> np.ndarray:
    with Image.open(path) as image:
        categorical = np.asarray(image)
    return categorical != background_value
