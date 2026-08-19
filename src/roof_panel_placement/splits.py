"""Development splits that cannot expose the official RID2 final test set."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Rid2Paths, load_roof_mask


def _stable_key(image_name: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{image_name}".encode()).hexdigest()


def _read_training_names(training_split: Path) -> list[str]:
    if training_split.name != "training_split_512.csv":
        raise RuntimeError("Method development may only use training_split_512.csv.")
    table = pd.read_csv(training_split)
    return table["image_names"].astype(str).tolist()


def build_development_manifest(
    paths: Rid2Paths,
    validation_fraction: float,
    seed: int,
    area_strata: list[float],
    background_value: int = 5,
) -> pd.DataFrame:
    """Create a deterministic, roof-area-stratified train/validation manifest."""
    names = _read_training_names(paths.training_split)
    rows = []
    for image_name in names:
        mask = load_roof_mask(paths.roof_mask_path(image_name), background_value)
        rows.append(
            {
                "image_name": image_name,
                "sample_id": Path(image_name).stem,
                "roof_fraction": float(mask.mean()),
                "stable_key": _stable_key(image_name, seed),
            }
        )

    manifest = pd.DataFrame(rows)
    boundaries = sorted(set(float(value) for value in area_strata))
    if boundaries[0] != 0.0 or boundaries[-1] != 1.0:
        raise ValueError("area_strata must begin at 0.0 and end at 1.0")
    manifest["area_stratum"] = "empty"
    positive = manifest["roof_fraction"] > 0
    manifest.loc[positive, "area_stratum"] = pd.cut(
        manifest.loc[positive, "roof_fraction"],
        bins=boundaries,
        include_lowest=False,
        duplicates="drop",
    ).astype(str)

    manifest["development_split"] = "train"
    for _, indices in manifest.groupby("area_stratum", observed=True).groups.items():
        ordered = manifest.loc[list(indices)].sort_values("stable_key")
        validation_count = max(1, round(len(ordered) * validation_fraction))
        manifest.loc[ordered.index[:validation_count], "development_split"] = "validation"

    return manifest.drop(columns="stable_key").sort_values("image_name").reset_index(drop=True)


def attach_runtime_paths(manifest: pd.DataFrame, paths: Rid2Paths) -> pd.DataFrame:
    """Attach ephemeral absolute paths without saving them to portable manifests."""
    result = manifest.copy()
    result["image_path"] = result["image_name"].map(paths.image_path)
    result["roof_mask_path"] = result["image_name"].map(paths.roof_mask_path)
    return result
