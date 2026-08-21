# Roof U-Net v2 medium-run checkpoint

Checkpoint date: 2026-08-21

Run ID: `roof_unet_v2_medium_20260821T105827Z`

The run used 400 fitting images, 100 inner-diagnostic images, and 100 outer
development/validation images. It accessed zero official final-test rows. The
downloadable run bundle and best weights are stored under
`/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/`.

## Training outcome

- Early stopping occurred after epoch 28.
- The lowest inner-diagnostic loss occurred at epoch 21 and determined the
  restored checkpoint.
- Pixel-aggregate inner-diagnostic IoU continued as high as 0.531061 at epoch
  28, illustrating that loss-selected and IoU-selected checkpoints can differ.
- Clean-image mean IoU was 0.459071 on sampled fitting images and 0.457234 on
  inner-diagnostic images.
- Clean inner-diagnostic mean absolute area error was 226.038528 m².

The small clean fit–diagnostic gap does not indicate classical overfitting.
Instead, the model remains biased toward excessive roof prediction: clean
inner-diagnostic recall was 0.864396 while precision was 0.512936.

## Outer-validation outcome

The area-aware policy selected the upper search boundary, threshold 0.70:

- Mean IoU: 0.459876
- Median IoU: 0.509992
- Mean absolute area error: 184.643776 m²

IoU increased and area error decreased monotonically across every tested
threshold from 0.30 through 0.70. The true operating optimum therefore was not
bracketed by the v2 search grid. This is evidence of poor probability
calibration and persistent over-segmentation, not evidence that 0.70 is a
stable final threshold.

## Interpretation

U-Net v2 underperformed both the v1 U-Net medium operating point (mean IoU
0.477774 and area error 117.879680 m² at threshold 0.50) and the SLIC–RF area
baseline (mean IoU 0.474801 and area error 110.190464 m²). GroupNorm plus the
combined photometric augmentation and scheduler changes did not improve the
current task. Because the factors were changed together, this run does not
identify which individual change caused the regression.

The run remains useful as a negative result and must not replace the saved v1
checkpoint.

