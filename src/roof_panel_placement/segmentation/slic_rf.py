"""Classical multiscale SLIC and Random Forest roof segmentation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage import color, filters, morphology, segmentation
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier


LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2


def slic_labels(image: np.ndarray, n_segments: int, config: dict) -> np.ndarray:
    """Create one compact SLIC partition with labels beginning at zero."""
    return segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=float(config["compactness"]),
        sigma=float(config["sigma"]),
        start_label=0,
        channel_axis=-1,
    )


def _pixel_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous features and uniform-LBP codes for every pixel."""
    rgb = image.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb).astype(np.float32)
    hsv = color.rgb2hsv(rgb).astype(np.float32)
    gray = color.rgb2gray(rgb).astype(np.float32)
    edge = filters.sobel(gray).astype(np.float32)
    local_mean = uniform_filter(gray, size=7)
    local_square_mean = uniform_filter(gray * gray, size=7)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))

    continuous = np.dstack((rgb, lab, hsv, gray, edge, local_std))
    gray_u8 = np.round(gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(
        gray_u8,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    ).astype(np.int16)
    return continuous, lbp


def superpixel_features(
    image: np.ndarray,
    labels: np.ndarray,
    prepared_pixels: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Summarize colour, texture, and edge evidence in every superpixel."""
    if prepared_pixels is None:
        prepared_pixels = _pixel_features(image)
    continuous, lbp = prepared_pixels
    flat_labels = labels.ravel()
    region_count = int(flat_labels.max()) + 1
    counts = np.bincount(flat_labels, minlength=region_count).astype(np.float64)

    summaries = []
    for channel in range(continuous.shape[-1]):
        values = continuous[..., channel].ravel().astype(np.float64)
        total = np.bincount(flat_labels, weights=values, minlength=region_count)
        square_total = np.bincount(
            flat_labels,
            weights=values * values,
            minlength=region_count,
        )
        mean = total / counts
        standard_deviation = np.sqrt(np.maximum(square_total / counts - mean * mean, 0.0))
        summaries.extend((mean, standard_deviation))

    lbp_histogram = np.zeros((region_count, LBP_BINS), dtype=np.float64)
    combined = flat_labels * LBP_BINS + lbp.ravel()
    histogram = np.bincount(combined, minlength=region_count * LBP_BINS)
    lbp_histogram[:] = histogram.reshape(region_count, LBP_BINS) / counts[:, None]

    log_area_fraction = np.log1p(counts) / np.log1p(labels.size)
    return np.column_stack((*summaries, lbp_histogram, log_area_fraction)).astype(np.float32)


def superpixel_roof_fraction(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    flat_labels = labels.ravel()
    region_count = int(flat_labels.max()) + 1
    counts = np.bincount(flat_labels, minlength=region_count)
    roof_counts = np.bincount(
        flat_labels,
        weights=np.asarray(reference, dtype=np.float32).ravel(),
        minlength=region_count,
    )
    return roof_counts / counts


def training_rows_for_image(
    image: np.ndarray,
    reference: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a balanced sample of unambiguous roof/non-roof superpixels."""
    rows = []
    targets = []
    feature_config = config["features"]
    per_class_limit = int(feature_config["max_rows_per_image_per_class"])
    prepared_pixels = _pixel_features(image)

    for n_segments in config["slic"]["n_segments"]:
        labels = slic_labels(image, int(n_segments), config["slic"])
        features = superpixel_features(image, labels, prepared_pixels)
        fractions = superpixel_roof_fraction(reference, labels)
        class_indices = {
            0: np.flatnonzero(fractions <= float(feature_config["negative_fraction"])),
            1: np.flatnonzero(fractions >= float(feature_config["positive_fraction"])),
        }
        for target, indices in class_indices.items():
            if len(indices) > per_class_limit:
                indices = rng.choice(indices, size=per_class_limit, replace=False)
            rows.append(features[indices])
            targets.append(np.full(len(indices), target, dtype=np.uint8))

    return np.vstack(rows), np.concatenate(targets)


def cap_balanced_training_rows(
    rows: np.ndarray,
    targets: np.ndarray,
    max_rows_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound Random Forest memory while retaining equal class opportunity."""
    selected = []
    for target in (0, 1):
        indices = np.flatnonzero(targets == target)
        if len(indices) > max_rows_per_class:
            indices = rng.choice(indices, size=max_rows_per_class, replace=False)
        selected.append(indices)
    selected_indices = np.concatenate(selected)
    rng.shuffle(selected_indices)
    return rows[selected_indices], targets[selected_indices]


def fit_random_forest(rows: np.ndarray, targets: np.ndarray, config: dict) -> RandomForestClassifier:
    forest_config = config["random_forest"]
    model = RandomForestClassifier(
        n_estimators=int(forest_config["n_estimators"]),
        max_depth=int(forest_config["max_depth"]),
        min_samples_leaf=int(forest_config["min_samples_leaf"]),
        max_features=forest_config["max_features"],
        class_weight="balanced_subsample",
        random_state=int(config["run"]["seed"]),
        n_jobs=int(forest_config["n_jobs"]),
    )
    return model.fit(rows, targets)


def predict_multiscale_probability(
    model: RandomForestClassifier,
    image: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Predict at every SLIC scale and return their pixelwise median."""
    probability_maps = {}
    prepared_pixels = _pixel_features(image)
    for n_segments in config["slic"]["n_segments"]:
        n_segments = int(n_segments)
        labels = slic_labels(image, n_segments, config["slic"])
        features = superpixel_features(image, labels, prepared_pixels)
        region_probability = model.predict_proba(features)[:, 1]
        probability_maps[n_segments] = region_probability[labels]

    stacked = np.stack(list(probability_maps.values()))
    return np.median(stacked, axis=0), probability_maps


def postprocess_probability(probability: np.ndarray, config: dict) -> np.ndarray:
    """Convert probabilities to a cleaned binary mask, including empty output."""
    post = config["postprocessing"]
    mask = probability >= float(post["probability_threshold"])
    radius = int(post["closing_radius_pixels"])
    if radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(radius))
    mask = morphology.remove_small_holes(
        mask,
        area_threshold=int(post["maximum_hole_pixels"]),
    )
    return morphology.remove_small_objects(
        mask,
        min_size=int(post["minimum_component_pixels"]),
    )


def pairwise_scale_stability(probability_maps: dict[int, np.ndarray], threshold: float) -> float:
    """Mean pairwise mask IoU across SLIC resolutions."""
    masks = [probability >= threshold for probability in probability_maps.values()]
    scores = []
    for first, second in combinations(masks, 2):
        union = np.logical_or(first, second).sum()
        if union:
            scores.append(float(np.logical_and(first, second).sum() / union))
        elif not first.any() and not second.any():
            scores.append(1.0)
    return float(np.mean(scores)) if scores else np.nan
