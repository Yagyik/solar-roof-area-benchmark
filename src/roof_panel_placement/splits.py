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


def stratified_subset(
    table: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    """Select exactly ``maximum`` cases while approximately preserving area strata."""
    if maximum is None or len(table) <= maximum:
        return table.reset_index(drop=True)
    if maximum <= 0:
        raise ValueError("maximum must be positive or None")

    groups = {
        str(name): group.copy()
        for name, group in table.groupby("area_stratum", observed=True, sort=True)
    }
    counts = {name: 0 for name in groups}
    if maximum >= len(groups):
        counts = {name: 1 for name in groups}
    desired = {name: maximum * len(group) / len(table) for name, group in groups.items()}

    while sum(counts.values()) < maximum:
        candidates = [name for name, group in groups.items() if counts[name] < len(group)]
        selected = max(candidates, key=lambda name: (desired[name] - counts[name], name))
        counts[selected] += 1

    pieces = []
    for name, group in groups.items():
        ordered = group.assign(
            subset_key=group["image_name"].map(lambda value: _stable_key(value, seed))
        ).sort_values("subset_key")
        pieces.append(ordered.iloc[: counts[name]].drop(columns="subset_key"))
    result = pd.concat(pieces)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_inner_diagnostics(
    training_cases: pd.DataFrame,
    diagnostic_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out complete training images for pre-validation diagnostics."""
    table = training_cases.copy()
    table["diagnostic_key"] = table["image_name"].map(lambda name: _stable_key(name, seed))
    diagnostic_indices = []
    for _, indices in table.groupby("area_stratum", observed=True).groups.items():
        ordered = table.loc[list(indices)].sort_values("diagnostic_key")
        if len(ordered) < 2:
            continue
        count = max(1, round(len(ordered) * diagnostic_fraction))
        diagnostic_indices.extend(ordered.index[: min(count, len(ordered) - 1)])

    diagnostic = table.loc[diagnostic_indices].drop(columns="diagnostic_key")
    fitting = table.drop(index=diagnostic_indices).drop(columns="diagnostic_key")
    return fitting.reset_index(drop=True), diagnostic.reset_index(drop=True)
