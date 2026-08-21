"""Binary segmentation and plan-view area metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _box_array(boxes: object) -> np.ndarray:
    """Return xyxy boxes as a consistently shaped floating-point array."""
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    return array.reshape(-1, 4)


def pairwise_box_iou(boxes_a: object, boxes_b: object) -> np.ndarray:
    """Calculate pairwise IoU for two xyxy box collections."""
    boxes_a = _box_array(boxes_a)
    boxes_b = _box_array(boxes_b)
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0.0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0.0), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def box_detection_metrics(
    prediction_boxes: object,
    reference_boxes: object,
    iou_thresholds: tuple[float, ...] = (0.30, 0.50, 0.75),
) -> dict[str, float | int]:
    """Match predicted and oracle boxes greedily and report localization quality."""
    prediction_boxes = _box_array(prediction_boxes)
    reference_boxes = _box_array(reference_boxes)
    iou = pairwise_box_iou(prediction_boxes, reference_boxes)
    matches = []
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    for flat_index in np.argsort(iou, axis=None)[::-1]:
        prediction_index, reference_index = np.unravel_index(flat_index, iou.shape)
        score = float(iou[prediction_index, reference_index])
        if score <= 0:
            break
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(int(prediction_index))
        used_references.add(int(reference_index))
        matches.append(score)

    result: dict[str, float | int] = {
        "predicted_box_count": int(len(prediction_boxes)),
        "reference_box_count": int(len(reference_boxes)),
        "mean_matched_box_iou": float(np.mean(matches)) if matches else np.nan,
        "mean_best_reference_box_iou": (
            float(iou.max(axis=0).mean()) if len(prediction_boxes) and len(reference_boxes) else 0.0
        ),
    }
    for threshold in iou_thresholds:
        suffix = f"{int(round(100 * threshold)):02d}"
        true_positive = sum(score >= threshold for score in matches)
        precision = true_positive / len(prediction_boxes) if len(prediction_boxes) else np.nan
        recall = true_positive / len(reference_boxes) if len(reference_boxes) else np.nan
        result[f"box_true_positive_iou_{suffix}"] = int(true_positive)
        result[f"box_precision_iou_{suffix}"] = precision
        result[f"box_recall_iou_{suffix}"] = recall
        result[f"box_f1_iou_{suffix}"] = (
            2 * precision * recall / (precision + recall)
            if precision > 0 and recall > 0
            else 0.0
        )
    return result


def parent_box_coverage_metrics(
    parent_boxes: object,
    reference_boxes: object,
    reference_mask: np.ndarray,
    minimum_box_coverage: float = 0.80,
) -> dict[str, float | int]:
    """Measure whether parent boxes admit oracle roofs without judging parent tightness."""
    if not 0 <= minimum_box_coverage <= 1:
        raise ValueError("minimum_box_coverage must lie between zero and one.")
    parent_boxes = _box_array(parent_boxes)
    reference_boxes = _box_array(reference_boxes)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    height, width = reference_mask.shape
    parent_region = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in parent_boxes:
        left = int(np.clip(np.floor(x1), 0, width))
        top = int(np.clip(np.floor(y1), 0, height))
        right = int(np.clip(np.ceil(x2), left, width))
        bottom = int(np.clip(np.ceil(y2), top, height))
        parent_region[top:bottom, left:right] = True

    if len(parent_boxes) and len(reference_boxes):
        top_left = np.maximum(parent_boxes[:, None, :2], reference_boxes[None, :, :2])
        bottom_right = np.minimum(parent_boxes[:, None, 2:], reference_boxes[None, :, 2:])
        intersection_size = np.maximum(bottom_right - top_left, 0.0)
        intersection = intersection_size[..., 0] * intersection_size[..., 1]
        reference_area = np.prod(
            np.maximum(reference_boxes[:, 2:] - reference_boxes[:, :2], 0.0), axis=1
        )
        coverage = np.divide(
            intersection,
            reference_area[None, :],
            out=np.zeros_like(intersection),
            where=reference_area[None, :] > 0,
        ).max(axis=0)
    else:
        coverage = np.zeros(len(reference_boxes), dtype=np.float64)

    reference_pixels = int(reference_mask.sum())
    covered_reference_boxes = int((coverage >= minimum_box_coverage).sum())
    covered_roof_pixels = int(np.logical_and(parent_region, reference_mask).sum())
    return {
        "parent_box_count": int(len(parent_boxes)),
        "reference_box_count": int(len(reference_boxes)),
        "parent_candidate_area_fraction": float(parent_region.mean()),
        "covered_roof_pixels": covered_roof_pixels,
        "reference_roof_pixels": reference_pixels,
        "roof_pixel_coverage": (
            float(covered_roof_pixels / reference_pixels)
            if reference_pixels
            else np.nan
        ),
        "mean_oracle_box_coverage": float(coverage.mean()) if len(coverage) else np.nan,
        "covered_reference_boxes": covered_reference_boxes,
        "oracle_box_coverage_recall": (
            float(covered_reference_boxes / len(coverage)) if len(coverage) else np.nan
        ),
    }


def binary_roof_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    pixel_area_m2: float,
) -> dict[str, float | int | bool]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ.")

    true_positive = int(np.logical_and(prediction, reference).sum())
    false_positive = int(np.logical_and(prediction, ~reference).sum())
    false_negative = int(np.logical_and(~prediction, reference).sum())
    union = true_positive + false_positive + false_negative
    predicted_pixels = int(prediction.sum())
    reference_pixels = int(reference.sum())
    both_empty = predicted_pixels == 0 and reference_pixels == 0

    precision = true_positive / predicted_pixels if predicted_pixels else np.nan
    recall = true_positive / reference_pixels if reference_pixels else np.nan
    iou = true_positive / union if union else np.nan
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else np.nan

    predicted_area = predicted_pixels * pixel_area_m2
    reference_area = reference_pixels * pixel_area_m2
    if both_empty:
        selection_score = 1.0
    elif reference_pixels == 0:
        selection_score = 0.0
    else:
        selection_score = iou
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_pixels": predicted_pixels,
        "reference_pixels": reference_pixels,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "both_empty": both_empty,
        "selection_score": selection_score,
        "predicted_area_m2": predicted_area,
        "reference_area_m2": reference_area,
        "signed_area_error_m2": predicted_area - reference_area,
        "absolute_area_error_m2": abs(predicted_area - reference_area),
        "false_positive_area_m2": false_positive * pixel_area_m2,
    }


def aggregate_roof_metrics(per_image: pd.DataFrame) -> dict[str, float | int]:
    valid_iou = per_image["iou"].dropna()
    empty_reference = per_image["reference_pixels"] == 0
    return {
        "images": int(len(per_image)),
        "mean_iou_nonempty_union": float(valid_iou.mean()) if len(valid_iou) else np.nan,
        "median_iou_nonempty_union": float(valid_iou.median()) if len(valid_iou) else np.nan,
        "mean_selection_score": float(per_image["selection_score"].mean()),
        "mean_absolute_area_error_m2": float(per_image["absolute_area_error_m2"].mean()),
        "empty_reference_images": int(empty_reference.sum()),
        "correctly_empty_images": int(per_image.loc[empty_reference, "both_empty"].sum()),
        "false_positive_area_on_empty_m2": float(
            per_image.loc[empty_reference, "false_positive_area_m2"].sum()
        ),
    }


def select_area_aware_candidate(
    candidates: pd.DataFrame,
    candidate_column: str,
    iou_tolerance: float,
) -> tuple[object, pd.DataFrame]:
    """Prefer minimum area error among candidates with near-best mean IoU."""
    required = {
        candidate_column,
        "mean_iou_nonempty_union",
        "mean_absolute_area_error_m2",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if iou_tolerance < 0:
        raise ValueError("iou_tolerance must be non-negative.")

    annotated = candidates.copy()
    finite_iou = annotated["mean_iou_nonempty_union"].dropna()
    if finite_iou.empty:
        raise ValueError("Candidate table contains no finite mean-IoU values.")
    best_iou = float(finite_iou.max())
    annotated["within_iou_tolerance"] = (
        annotated["mean_iou_nonempty_union"] >= best_iou - iou_tolerance
    )
    eligible = annotated.loc[annotated["within_iou_tolerance"]].sort_values(
        ["mean_absolute_area_error_m2", "mean_iou_nonempty_union"],
        ascending=[True, False],
    )
    selected_index = eligible.index[0]
    annotated["selected"] = annotated.index == selected_index
    selected_candidate = annotated.loc[selected_index, candidate_column]
    return selected_candidate, annotated


def select_area_aware_operating_point(
    sensitivity: pd.DataFrame,
    iou_tolerance: float,
) -> tuple[float, pd.DataFrame]:
    """Prefer minimum area error among thresholds with near-best mean IoU."""
    selected_threshold, annotated = select_area_aware_candidate(
        sensitivity,
        "probability_threshold",
        iou_tolerance,
    )
    selected_threshold = float(selected_threshold)
    return selected_threshold, annotated
