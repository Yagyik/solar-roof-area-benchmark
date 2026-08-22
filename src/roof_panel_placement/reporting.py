"""Deterministic cohorts and summary tables for roof-segmentation reports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


ROOF_COVERAGE_CLASSES = (
    "<10%",
    "10–35%",
    "35–70%",
    ">70%",
)


def roof_coverage_class(roof_fraction: float) -> str:
    """Assign one exhaustive class with 10%, 35%, and 70% as explicit boundaries."""
    value = float(roof_fraction)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Roof fraction must be finite and within [0, 1]; found {value}.")
    if value < 0.10:
        return ROOF_COVERAGE_CLASSES[0]
    if value < 0.35:
        return ROOF_COVERAGE_CLASSES[1]
    if value <= 0.70:
        return ROOF_COVERAGE_CLASSES[2]
    return ROOF_COVERAGE_CLASSES[3]


def attach_roof_coverage_class(
    table: pd.DataFrame,
    fraction_column: str = "roof_fraction",
) -> pd.DataFrame:
    """Attach the ordered four-class reporting cohort to a case table."""
    if fraction_column not in table.columns:
        raise ValueError(f"Missing roof-fraction column: {fraction_column}")
    result = table.copy()
    labels = result[fraction_column].map(roof_coverage_class)
    result["roof_coverage_class"] = pd.Categorical(
        labels,
        categories=ROOF_COVERAGE_CLASSES,
        ordered=True,
    )
    return result


def validate_common_case_ids(method_tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Require every method table to contain the same unique validation IDs."""
    if not method_tables:
        raise ValueError("At least one method table is required.")
    expected_name = next(iter(method_tables))
    expected_table = method_tables[expected_name]
    expected_ids = expected_table["sample_id"].astype(str)
    if expected_ids.duplicated().any():
        raise ValueError(f"Method {expected_name} contains duplicate sample IDs.")
    expected = set(expected_ids)
    for method_name, table in method_tables.items():
        sample_ids = table["sample_id"].astype(str)
        if sample_ids.duplicated().any():
            raise ValueError(f"Method {method_name} contains duplicate sample IDs.")
        observed = set(sample_ids)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"Method {method_name} does not share the {expected_name} cases; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
    return sorted(expected)


def summarize_methods_by_coverage(
    method_tables: Mapping[str, pd.DataFrame],
    case_index: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate comparable mask and area statistics for every method and cohort."""
    common_ids = validate_common_case_ids(method_tables)
    index = case_index.copy()
    index["sample_id"] = index["sample_id"].astype(str)
    if index["sample_id"].duplicated().any():
        raise ValueError("Case index contains duplicate sample IDs.")
    if set(index["sample_id"]) != set(common_ids):
        raise ValueError("Case index and method tables do not contain identical sample IDs.")
    if "roof_coverage_class" not in index.columns:
        index = attach_roof_coverage_class(index)

    rows = []
    for method_name, table in method_tables.items():
        metrics = table.copy()
        metrics["sample_id"] = metrics["sample_id"].astype(str)
        metrics = metrics.merge(
            index[["sample_id", "roof_fraction", "roof_coverage_class"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        for coverage_class in ROOF_COVERAGE_CLASSES:
            group = metrics.loc[metrics["roof_coverage_class"] == coverage_class]
            rows.append(
                {
                    "method": method_name,
                    "roof_coverage_class": coverage_class,
                    "images": int(len(group)),
                    "mean_roof_fraction": float(group["roof_fraction"].mean()),
                    "mean_iou": float(group["iou"].mean()),
                    "median_iou": float(group["iou"].median()),
                    "mean_precision": float(group["precision"].mean()),
                    "mean_recall": float(group["recall"].mean()),
                    "mean_absolute_area_error_m2": float(
                        group["absolute_area_error_m2"].mean()
                    ),
                    "mean_signed_area_error_m2": float(
                        group["signed_area_error_m2"].mean()
                    ),
                    "iou_at_least_0_50_fraction": float((group["iou"] >= 0.50).mean()),
                }
            )
    result = pd.DataFrame(rows)
    result["roof_coverage_class"] = pd.Categorical(
        result["roof_coverage_class"],
        categories=ROOF_COVERAGE_CLASSES,
        ordered=True,
    )
    return result.sort_values(["roof_coverage_class", "method"]).reset_index(drop=True)


def select_median_coverage_examples(case_index: pd.DataFrame) -> pd.DataFrame:
    """Select one method-independent case nearest each cohort's median coverage."""
    index = case_index.copy()
    if "roof_coverage_class" not in index.columns:
        index = attach_roof_coverage_class(index)
    selected = []
    for coverage_class in ROOF_COVERAGE_CLASSES:
        group = index.loc[index["roof_coverage_class"] == coverage_class].copy()
        if group.empty:
            raise ValueError(f"No validation case is available for class {coverage_class}.")
        median_fraction = float(group["roof_fraction"].median())
        group["distance_from_class_median"] = (
            group["roof_fraction"] - median_fraction
        ).abs()
        chosen = group.sort_values(
            ["distance_from_class_median", "sample_id"]
        ).iloc[0].copy()
        chosen["class_median_roof_fraction"] = median_fraction
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def select_easy_high_roof_cases(
    slic_metrics: pd.DataFrame,
    case_index: pd.DataFrame,
    count: int = 4,
    minimum_roof_fraction: float = 0.40,
) -> pd.DataFrame:
    """Select explicitly pedagogical high-IoU SLIC cases above a roof-fraction floor."""
    metrics = slic_metrics.copy()
    metrics["sample_id"] = metrics["sample_id"].astype(str)
    index = case_index.copy()
    index["sample_id"] = index["sample_id"].astype(str)
    candidates = metrics.merge(
        index[["sample_id", "image_name", "roof_fraction"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    candidates = candidates.loc[
        candidates["roof_fraction"] > float(minimum_roof_fraction)
    ].sort_values(
        ["iou", "absolute_area_error_m2", "sample_id"],
        ascending=[False, True, True],
    )
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} cases exceed roof fraction {minimum_roof_fraction}."
        )
    result = candidates.head(count).copy()
    result.insert(0, "walkthrough_row", np.arange(1, count + 1))
    result["selection_reason"] = (
        f"Top SLIC IoU among validation cases with roof fraction >{minimum_roof_fraction:.0%}"
    )
    return result.reset_index(drop=True)
