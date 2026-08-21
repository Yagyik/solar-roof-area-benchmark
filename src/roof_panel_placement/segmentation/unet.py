"""Compact from-scratch U-Net for binary roof segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..data import load_rgb, load_roof_mask


class RoofDataset(Dataset):
    """Load RID2 image/mask pairs with simple paired geometric augmentation."""

    def __init__(
        self,
        cases,
        background_value: int,
        augment: bool = False,
        augmentation_config: dict | None = None,
    ) -> None:
        self.cases = cases.reset_index(drop=True)
        self.background_value = background_value
        self.augment = augment
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.cases.iloc[index]
        image = load_rgb(Path(row.image_path))
        mask = load_roof_mask(Path(row.roof_mask_path), self.background_value)

        if self.augment:
            rotation = int(torch.randint(0, 4, ()).item())
            image = np.rot90(image, rotation)
            mask = np.rot90(mask, rotation)
            if bool(torch.randint(0, 2, ()).item()):
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            if bool(torch.randint(0, 2, ()).item()):
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
            image = _photometric_augmentation(image, self.augmentation_config)

        image_tensor = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        mask_tensor = torch.from_numpy(np.array(mask, copy=True, order="C"))[None].float()
        return image_tensor, mask_tensor, str(row.sample_id)


def _random_symmetric(maximum: float) -> float:
    return float((2.0 * torch.rand(()) - 1.0).item()) * maximum


def _photometric_augmentation(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply mild RGB-only jitter while leaving the paired mask unchanged."""
    if not config:
        return image

    adjusted = np.asarray(image, dtype=np.float32) / 255.0
    brightness = float(config.get("brightness", 0.0))
    if brightness > 0:
        adjusted *= 1.0 + _random_symmetric(brightness)

    contrast = float(config.get("contrast", 0.0))
    if contrast > 0:
        channel_mean = adjusted.mean(axis=(0, 1), keepdims=True)
        adjusted = channel_mean + (adjusted - channel_mean) * (
            1.0 + _random_symmetric(contrast)
        )

    saturation = float(config.get("saturation", 0.0))
    if saturation > 0:
        gray = adjusted.mean(axis=2, keepdims=True)
        adjusted = gray + (adjusted - gray) * (1.0 + _random_symmetric(saturation))

    noise_standard_deviation = float(config.get("noise_standard_deviation", 0.0))
    if noise_standard_deviation > 0:
        noise = torch.randn(adjusted.shape).numpy() * noise_standard_deviation
        adjusted += noise.astype(np.float32)

    return np.round(np.clip(adjusted, 0.0, 1.0) * 255.0).astype(np.uint8)


def _normalization_layer(kind: str, channels: int, group_count: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        if group_count <= 0:
            raise ValueError("group_count must be positive.")
        groups = min(group_count, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization: {kind}")


class DoubleConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        normalization: str = "batch",
        group_count: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            _normalization_layer(normalization, output_channels, group_count),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            _normalization_layer(normalization, output_channels, group_count),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        normalization: str = "batch",
        group_count: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, output_channels, normalization, group_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        normalization: str = "batch",
        group_count: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, 2, stride=2)
        self.conv = DoubleConv(
            output_channels + skip_channels,
            output_channels,
            normalization,
            group_count,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        return self.conv(torch.cat((skip, inputs), dim=1))


class RoofUNet(nn.Module):
    """Four-level U-Net that preserves the 512×512 output resolution."""

    def __init__(
        self,
        base_channels: int = 32,
        normalization: str = "batch",
        group_count: int = 8,
    ) -> None:
        super().__init__()
        channels = [base_channels * 2**level for level in range(5)]
        self.input_block = DoubleConv(3, channels[0], normalization, group_count)
        self.down_blocks = nn.ModuleList(
            DownBlock(
                channels[level],
                channels[level + 1],
                normalization,
                group_count,
            )
            for level in range(4)
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(
                channels[level],
                channels[level - 1],
                channels[level - 1],
                normalization,
                group_count,
            )
            for level in range(4, 0, -1)
        )
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [self.input_block(inputs)]
        for block in self.down_blocks:
            features.append(block(features[-1]))
        decoded = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            decoded = block(decoded, skip)
        return self.output(decoded)


class BCEDiceLoss(nn.Module):
    def __init__(self, positive_weight: float, dice_weight: float) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_weight,
        )
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


def estimate_positive_weight(cases, background_value: int, maximum: float) -> float:
    """Calculate a bounded negative-to-positive pixel ratio from fitting masks."""
    positive = 0
    total = 0
    for row in cases.itertuples(index=False):
        mask = load_roof_mask(Path(row.roof_mask_path), background_value)
        positive += int(mask.sum())
        total += int(mask.size)
    if positive == 0:
        raise ValueError("Fitting masks contain no positive roof pixels.")
    return float(np.clip((total - positive) / positive, 1.0, maximum))


def _batch_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    prediction = torch.sigmoid(logits) >= threshold
    reference = targets >= 0.5
    return {
        "true_positive": int(torch.logical_and(prediction, reference).sum().item()),
        "false_positive": int(torch.logical_and(prediction, ~reference).sum().item()),
        "false_negative": int(torch.logical_and(~prediction, reference).sum().item()),
    }


def _epoch_metrics(loss_sum: float, batches: int, counts: dict) -> dict[str, float]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    union = true_positive + false_positive + false_negative
    return {
        "loss": loss_sum / batches,
        "iou": true_positive / union if union else 1.0,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "dice": (2 * true_positive)
        / (2 * true_positive + false_positive + false_negative)
        if 2 * true_positive + false_positive + false_negative
        else 1.0,
    }


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer=None,
    scaler=None,
) -> dict[str, float]:
    """Run one fitting or evaluation epoch and aggregate pixelwise metrics."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.detach().item())
            batch_counts = _batch_counts(logits.detach(), masks, threshold)
            for key in counts:
                counts[key] += batch_counts[key]

    return _epoch_metrics(loss_sum, len(loader), counts)


def fit_unet(
    model: nn.Module,
    fitting_loader,
    diagnostic_loader,
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> list[dict[str, float | int | str]]:
    """Fit with held-image diagnostics, early stopping, and a persistent best checkpoint."""
    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = BCEDiceLoss(
        positive_weight=float(training_config["positive_weight"]),
        dice_weight=float(training_config["dice_weight"]),
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = None
    if "scheduler_factor" in training_config:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_config["scheduler_factor"]),
            patience=int(training_config["scheduler_patience"]),
            min_lr=float(training_config["minimum_learning_rate"]),
        )
    history = []
    best_loss = float("inf")
    checkpoint_written = False
    stale_epochs = 0

    for epoch in range(1, int(training_config["epochs"]) + 1):
        fit_metrics = run_epoch(
            model,
            fitting_loader,
            criterion,
            device,
            float(config["inference"]["probability_threshold"]),
            optimizer=optimizer,
            scaler=scaler,
        )
        diagnostic_metrics = run_epoch(
            model,
            diagnostic_loader,
            criterion,
            device,
            float(config["inference"]["probability_threshold"]),
        )
        for dataset_name, metrics in (
            ("fit", fit_metrics),
            ("inner_diagnostic", diagnostic_metrics),
        ):
            history.append(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **metrics,
                }
            )
        print(
            f"Epoch {epoch:02d}: fit loss={fit_metrics['loss']:.4f}, "
            f"diagnostic loss={diagnostic_metrics['loss']:.4f}, "
            f"diagnostic IoU={diagnostic_metrics['iou']:.4f}"
        )

        if scheduler is not None:
            scheduler.step(diagnostic_metrics["loss"])

        if diagnostic_metrics["loss"] < best_loss:
            best_loss = diagnostic_metrics["loss"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_written = True
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_config["early_stopping_patience"]):
                print(f"Early stopping after epoch {epoch}.")
                break

    if not checkpoint_written:
        raise RuntimeError("Training completed without producing a checkpoint.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return history


@torch.inference_mode()
def predict_probability(model: nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict a full-resolution roof probability map for one RGB image."""
    inputs = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1)[None].float()
    inputs = inputs.to(device) / 255.0
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        probability = torch.sigmoid(model(inputs))[0, 0]
    return probability.float().cpu().numpy()
