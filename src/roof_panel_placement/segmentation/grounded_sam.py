"""Grounding DINO box prompts followed by SAM 2.1 roof masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision.ops import nms
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


@dataclass
class GroundedSamModels:
    detector_processor: object
    detector: object
    segmenter_processor: object
    segmenter: object
    device: torch.device


def load_models(config: dict, device: torch.device) -> GroundedSamModels:
    """Load pinned open checkpoints, using a persistent Hugging Face cache."""
    model_config = config["models"]
    cache_directory = config["paths"]["model_cache"]
    detector_processor = AutoProcessor.from_pretrained(
        model_config["detector_id"],
        revision=model_config["detector_revision"],
        cache_dir=cache_directory,
        token=False,
    )
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_config["detector_id"],
        revision=model_config["detector_revision"],
        cache_dir=cache_directory,
        use_safetensors=True,
        token=False,
    ).to(device).eval()
    segmenter_processor = Sam2Processor.from_pretrained(
        model_config["segmenter_id"],
        revision=model_config["segmenter_revision"],
        cache_dir=cache_directory,
        token=False,
    )
    segmenter = Sam2Model.from_pretrained(
        model_config["segmenter_id"],
        revision=model_config["segmenter_revision"],
        cache_dir=cache_directory,
        use_safetensors=True,
        token=False,
    ).to(device).eval()
    return GroundedSamModels(
        detector_processor=detector_processor,
        detector=detector,
        segmenter_processor=segmenter_processor,
        segmenter=segmenter,
        device=device,
    )


def _pil_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


@torch.inference_mode()
def detect_boxes(
    models: GroundedSamModels,
    image: np.ndarray,
    prompts: list[str],
    box_threshold: float,
    text_threshold: float,
    maximum_boxes: int,
) -> dict:
    """Detect all prompt-matched boxes in one image."""
    if not prompts:
        raise ValueError("At least one text prompt is required.")
    pil_image = _pil_image(image)
    inputs = models.detector_processor(
        images=pil_image,
        text=[prompts],
        return_tensors="pt",
    ).to(models.device)
    with torch.autocast(device_type="cuda", enabled=models.device.type == "cuda"):
        outputs = models.detector(**inputs)
    result = models.detector_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.shape[:2]],
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    boxes = result["boxes"].detach().float().cpu()
    scores = result["scores"].detach().float().cpu()
    order = torch.argsort(scores, descending=True)[:maximum_boxes]
    return {
        "boxes": boxes[order],
        "scores": scores[order],
        "labels": [str(labels[index]) for index in order.tolist()],
    }


def _deduplicate_detections(detections: list[dict], iou_threshold: float) -> dict:
    nonempty = [item for item in detections if len(item["boxes"])]
    if not nonempty:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty(0, dtype=torch.float32),
            "labels": [],
        }
    boxes = torch.cat([item["boxes"] for item in nonempty])
    scores = torch.cat([item["scores"] for item in nonempty])
    labels = [label for item in nonempty for label in item["labels"]]
    keep = nms(boxes, scores, iou_threshold).tolist()
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": [labels[index] for index in keep],
    }


def _limit_detections(detection: dict, maximum_boxes: int) -> dict:
    """Keep the highest-scoring detections after cross-crop deduplication."""
    if maximum_boxes <= 0:
        raise ValueError("maximum_boxes must be positive.")
    order = torch.argsort(detection["scores"], descending=True)[:maximum_boxes]
    return {
        "boxes": detection["boxes"][order],
        "scores": detection["scores"][order],
        "labels": [detection["labels"][index] for index in order.tolist()],
    }


def _expand_boxes(boxes: torch.Tensor, image_shape: tuple[int, int], fraction: float) -> torch.Tensor:
    if not len(boxes) or fraction <= 0:
        return boxes
    height, width = image_shape
    expanded = boxes.clone()
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    expanded[:, 0] -= fraction * box_width
    expanded[:, 2] += fraction * box_width
    expanded[:, 1] -= fraction * box_height
    expanded[:, 3] += fraction * box_height
    expanded[:, [0, 2]] = expanded[:, [0, 2]].clamp(0, width - 1)
    expanded[:, [1, 3]] = expanded[:, [1, 3]].clamp(0, height - 1)
    return expanded


@torch.inference_mode()
def segment_boxes(
    models: GroundedSamModels,
    image: np.ndarray,
    boxes: torch.Tensor,
    minimum_predicted_iou: float,
    box_padding_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM 2.1 once for all detector boxes and return masks and mask scores."""
    height, width = image.shape[:2]
    if not len(boxes):
        return np.empty((0, height, width), dtype=bool), np.empty(0, dtype=np.float32)
    boxes = _expand_boxes(boxes, (height, width), box_padding_fraction)
    inputs = models.segmenter_processor(
        images=_pil_image(image),
        input_boxes=[boxes.tolist()],
        return_tensors="pt",
    ).to(models.device)
    with torch.autocast(device_type="cuda", enabled=models.device.type == "cuda"):
        outputs = models.segmenter(**inputs, multimask_output=False)
    masks = models.segmenter_processor.post_process_masks(
        outputs.pred_masks.float().cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    if masks.ndim == 4:
        masks = masks[:, 0]
    scores = outputs.iou_scores.detach().float().cpu().reshape(-1)
    keep = scores >= minimum_predicted_iou
    return masks[keep].numpy().astype(bool), scores[keep].numpy()


def _mask_from_detection(
    models: GroundedSamModels,
    image: np.ndarray,
    detection: dict,
    config: dict,
) -> dict:
    masks, mask_scores = segment_boxes(
        models,
        image,
        detection["boxes"],
        float(config["segmentation"]["minimum_predicted_iou"]),
        float(config["segmentation"]["box_padding_fraction"]),
    )
    combined = np.logical_or.reduce(masks, axis=0) if len(masks) else np.zeros(image.shape[:2], bool)
    return {
        "mask": combined,
        **detection,
        "sam_scores": mask_scores,
        "instance_masks": masks,
    }


def predict_joint(
    models: GroundedSamModels,
    image: np.ndarray,
    prompts: list[str],
    config: dict,
) -> dict:
    detection = detect_boxes(
        models,
        image,
        prompts,
        float(config["detection"]["box_threshold"]),
        float(config["detection"]["text_threshold"]),
        int(config["detection"]["maximum_boxes"]),
    )
    return _mask_from_detection(models, image, detection, config)


def predict_independent(
    models: GroundedSamModels,
    image: np.ndarray,
    prompts: list[str],
    config: dict,
    consensus_votes: int,
) -> dict:
    """Run one detector call per prompt, then union or vote over prompt masks."""
    prompt_results = []
    for prompt in prompts:
        detection = detect_boxes(
            models,
            image,
            [prompt],
            float(config["detection"]["box_threshold"]),
            float(config["detection"]["text_threshold"]),
            int(config["detection"]["maximum_boxes"]),
        )
        prompt_results.append(_mask_from_detection(models, image, detection, config))
    votes = np.stack([result["mask"] for result in prompt_results]).sum(axis=0)
    detections = _deduplicate_detections(
        prompt_results,
        float(config["detection"]["nms_iou_threshold"]),
    )
    return {
        "mask": votes >= consensus_votes,
        **detections,
        "sam_scores": np.concatenate(
            [result["sam_scores"] for result in prompt_results if len(result["sam_scores"])]
        )
        if any(len(result["sam_scores"]) for result in prompt_results)
        else np.empty(0, dtype=np.float32),
        "prompt_masks": [result["mask"] for result in prompt_results],
    }


def detect_parent_candidates(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    config: dict,
) -> dict:
    """Generate the raw scored parent pool before optional parent NMS."""
    detection_config = config["detection"]
    maximum_parent_boxes = int(detection_config["maximum_parent_boxes"])
    parent_nms_threshold = detection_config.get("parent_nms_iou_threshold")
    maximum_raw_boxes = (
        int(detection_config["maximum_raw_parent_boxes"])
        if parent_nms_threshold is not None
        else maximum_parent_boxes
    )
    candidates = detect_boxes(
        models,
        image,
        parent_prompts,
        float(detection_config["parent_box_threshold"]),
        float(detection_config["text_threshold"]),
        maximum_raw_boxes,
    )
    return {
        **candidates,
        "raw_box_count": len(candidates["boxes"]),
        "raw_cap_reached": len(candidates["boxes"]) == maximum_raw_boxes,
    }


def apply_parent_nms(
    candidates: dict,
    iou_threshold: float,
    maximum_parent_boxes: int,
) -> dict:
    """Apply class-agnostic NMS to a raw parent pool, then enforce the final cap."""
    if not len(candidates["boxes"]):
        return {
            **candidates,
            "nms_box_count": 0,
            "suppressed_box_count": 0,
            "final_cap_reached": False,
        }
    keep_after_nms = nms(
        candidates["boxes"],
        candidates["scores"],
        float(iou_threshold),
    ).tolist()
    nms_box_count = len(keep_after_nms)
    keep = keep_after_nms[:maximum_parent_boxes]
    return {
        "boxes": candidates["boxes"][keep],
        "scores": candidates["scores"][keep],
        "labels": [candidates["labels"][index] for index in keep],
        "raw_box_count": int(candidates["raw_box_count"]),
        "raw_cap_reached": bool(candidates["raw_cap_reached"]),
        "nms_box_count": nms_box_count,
        "suppressed_box_count": len(candidates["boxes"]) - nms_box_count,
        "final_cap_reached": nms_box_count > maximum_parent_boxes,
    }


def detect_parent_boxes(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    config: dict,
) -> dict:
    """Detect parents, optionally applying class-agnostic NMS before the final cap."""
    detection_config = config["detection"]
    maximum_parent_boxes = int(detection_config["maximum_parent_boxes"])
    candidates = detect_parent_candidates(models, image, parent_prompts, config)
    parent_nms_threshold = detection_config.get("parent_nms_iou_threshold")
    if parent_nms_threshold is None:
        return {
            **candidates,
            "nms_box_count": len(candidates["boxes"]),
            "suppressed_box_count": 0,
            "final_cap_reached": len(candidates["boxes"]) == maximum_parent_boxes,
        }
    return apply_parent_nms(
        candidates,
        float(parent_nms_threshold),
        maximum_parent_boxes,
    )


def _predict_hierarchical_from_parents(
    models: GroundedSamModels,
    image: np.ndarray,
    child_prompts: list[str],
    parent: dict,
    config: dict,
    maximum_output_boxes: int | None = None,
) -> dict:
    child_detections = []
    height, width = image.shape[:2]
    for parent_box in parent["boxes"]:
        x1, y1, x2, y2 = parent_box.round().to(torch.int64).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
        crop = image[y1:y2, x1:x2]
        child = detect_boxes(
            models,
            crop,
            child_prompts,
            float(config["detection"]["box_threshold"]),
            float(config["detection"]["text_threshold"]),
            int(config["detection"]["maximum_boxes"]),
        )
        if len(child["boxes"]):
            translated_boxes = child["boxes"].clone()
            translated_boxes[:, [0, 2]] += x1
            translated_boxes[:, [1, 3]] += y1
            child = {**child, "boxes": translated_boxes}
            child_detections.append(child)
    children = _deduplicate_detections(
        child_detections,
        float(config["detection"]["nms_iou_threshold"]),
    )
    if maximum_output_boxes is not None:
        children = _limit_detections(children, maximum_output_boxes)
    result = _mask_from_detection(models, image, children, config)
    result["parent_box_count"] = len(parent["boxes"])
    result["parent_boxes"] = parent["boxes"]
    result["parent_scores"] = parent["scores"]
    result["parent_labels"] = parent["labels"]
    for key in (
        "raw_box_count",
        "raw_cap_reached",
        "nms_box_count",
        "suppressed_box_count",
        "final_cap_reached",
    ):
        if key in parent:
            result[f"parent_{key}"] = parent[key]
    return result


def predict_hierarchical(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    child_prompts: list[str],
    config: dict,
) -> dict:
    """Detect buildings first, then search each building crop for roof boxes."""
    parent = detect_parent_boxes(models, image, parent_prompts, config)
    global_limit = config["detection"].get("global_maximum_boxes")
    return _predict_hierarchical_from_parents(
        models,
        image,
        child_prompts,
        parent,
        config,
        maximum_output_boxes=int(global_limit) if global_limit is not None else None,
    )


def predict_hierarchical_capped(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    child_prompts: list[str],
    config: dict,
) -> dict:
    """Run the un-clipped hierarchy with a global final roof-box cap."""
    parent = detect_parent_boxes(models, image, parent_prompts, config)
    return _predict_hierarchical_from_parents(
        models,
        image,
        child_prompts,
        parent,
        config,
        maximum_output_boxes=int(config["detection"]["maximum_boxes"]),
    )


def predict_parent_box_sam(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    config: dict,
) -> dict:
    """Detect buildings and send those parent boxes directly to SAM."""
    parent = detect_parent_boxes(models, image, parent_prompts, config)
    result = _mask_from_detection(models, image, parent, config)
    result["parent_box_count"] = len(parent["boxes"])
    return result


def predict_hierarchical_clipped(
    models: GroundedSamModels,
    image: np.ndarray,
    parent_prompts: list[str],
    child_prompts: list[str],
    config: dict,
) -> dict:
    """Intersect hierarchical roof masks with SAM masks from parent building boxes."""
    parent = detect_parent_boxes(models, image, parent_prompts, config)
    child_result = _predict_hierarchical_from_parents(
        models,
        image,
        child_prompts,
        parent,
        config,
        maximum_output_boxes=int(config["detection"]["maximum_boxes"]),
    )
    parent_result = _mask_from_detection(models, image, parent, config)
    unclipped_mask = child_result["mask"]
    return {
        **child_result,
        "mask": np.logical_and(unclipped_mask, parent_result["mask"]),
        "unclipped_mask": unclipped_mask,
        "parent_mask": parent_result["mask"],
        "parent_boxes": parent_result["boxes"],
        "parent_sam_scores": parent_result["sam_scores"],
    }


def oracle_component_boxes(
    reference: np.ndarray,
    minimum_component_pixels: int = 1,
) -> torch.Tensor:
    """Convert oracle connected components to boxes for a stage-isolation diagnostic."""
    if minimum_component_pixels <= 0:
        raise ValueError("minimum_component_pixels must be positive.")
    labels, component_count = ndimage.label(np.asarray(reference, dtype=bool))
    boxes = []
    for component_id, slices in enumerate(
        ndimage.find_objects(labels, max_label=component_count),
        start=1,
    ):
        if slices is None:
            continue
        component_pixels = int((labels[slices] == component_id).sum())
        if component_pixels < minimum_component_pixels:
            continue
        y_slice, x_slice = slices
        boxes.append(
            [
                float(x_slice.start),
                float(y_slice.start),
                float(x_slice.stop),
                float(y_slice.stop),
            ]
        )
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def predict_oracle_box_sam(
    models: GroundedSamModels,
    image: np.ndarray,
    reference: np.ndarray,
    config: dict,
) -> dict:
    """Measure SAM with oracle component boxes; never use this as a deployable method."""
    boxes = oracle_component_boxes(
        reference,
        int(config["diagnostics"]["minimum_oracle_component_pixels"]),
    )
    detection = {
        "boxes": boxes,
        "scores": torch.ones(len(boxes), dtype=torch.float32),
        "labels": ["oracle roof component"] * len(boxes),
    }
    return _mask_from_detection(models, image, detection, config)


def predict_strategy(
    models: GroundedSamModels,
    image: np.ndarray,
    strategy: str,
    config: dict,
) -> dict:
    prompts = list(config["prompts"]["roof"])
    if strategy == "joint":
        return predict_joint(models, image, prompts, config)
    if strategy == "independent_union":
        return predict_independent(models, image, prompts, config, consensus_votes=1)
    if strategy == "independent_consensus":
        return predict_independent(
            models,
            image,
            prompts,
            config,
            consensus_votes=int(config["prompts"]["consensus_votes"]),
        )
    if strategy == "hierarchical":
        return predict_hierarchical(
            models,
            image,
            list(config["prompts"]["parent"]),
            prompts,
            config,
        )
    if strategy == "hierarchical_capped":
        return predict_hierarchical_capped(
            models,
            image,
            list(config["prompts"]["parent"]),
            prompts,
            config,
        )
    if strategy == "hierarchical_clipped":
        return predict_hierarchical_clipped(
            models,
            image,
            list(config["prompts"]["parent"]),
            prompts,
            config,
        )
    if strategy == "parent_box_sam":
        return predict_parent_box_sam(
            models,
            image,
            list(config["prompts"]["parent"]),
            config,
        )
    raise ValueError(f"Unknown prompt strategy: {strategy}")
