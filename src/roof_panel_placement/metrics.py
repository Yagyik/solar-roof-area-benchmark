"""Binary segmentation and plan-view area metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
