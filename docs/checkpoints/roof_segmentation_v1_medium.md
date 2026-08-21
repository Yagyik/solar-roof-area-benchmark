# Roof segmentation v1 medium-run checkpoint

Checkpoint date: 2026-08-21

This checkpoint preserves the first informative medium-run comparison between
the classical multiscale SLIC–Random Forest and the from-scratch U-Net. Smoke
runs remain engineering checks and are not used to rank methods.

## Source and data controls

- Git revision used by the runs: `8625b61`
- Random seed: `20260819`
- Fitting images: 400
- Inner-diagnostic images: 100
- Outer-development/validation images: 100
- Official final-test rows accessed: 0
- Pixel size used for area conversion: 0.08 m

The saved development manifest and seed determine the cases. The run bundles
also contain the effective configuration and package environment.

## Persistent run bundles

### SLIC–Random Forest

- Run ID: `roof_slic_rf_medium_20260819T091131Z`
- Directory:
  `/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/roof_slic_rf_medium_20260819T091131Z`
- Downloadable archive: the sibling directory path with `.zip` appended
- Selected operating point: probability threshold 0.65, minimum component 250
  pixels

The bundle contains the fitted Random Forest, effective configuration,
development manifest, learning curve, mask diagnostics, post-processing
sensitivity, per-image and aggregate metrics, environment snapshot, and
representative figures.

### U-Net v1

- Run ID: `roof_unet_medium_20260821T092740Z`
- Directory:
  `/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/roof_unet_medium_20260821T092740Z`
- Downloadable archive: the sibling directory path with `.zip` appended
- Checkpoint: `model/best_model.pt`
- Mean-IoU-selected threshold: 0.35
- More area-balanced retrospective operating point: 0.50

The bundle contains the best model weights, effective configuration,
development manifest, complete training history, training-mask diagnostics,
threshold sensitivity, per-image and aggregate metrics, environment snapshot,
and representative figures.

The notebook reported successful creation of both Drive directories and ZIP
archives. The Drive files cannot be inspected from the local development
workspace, so downloading or listing the ZIP remains the final external-storage
integrity check.

## Recorded results

| Method and operating point | Mean IoU | Median IoU | Mean absolute area error |
|---|---:|---:|---:|
| SLIC–RF, threshold 0.65 | 0.474801 | 0.489471 | 110.190464 m² |
| U-Net v1, threshold 0.35 | 0.482294 | 0.500462 | 167.574784 m² |
| U-Net v1, threshold 0.50 | 0.477774 | 0.530910 | 117.879680 m² |
| U-Net v1, threshold 0.65 | 0.459271 | 0.511631 | 123.193024 m² |

The U-Net threshold selected solely by mean IoU gains only 0.007493 absolute
IoU over SLIC–RF while producing substantially worse area error. This motivates
the area-aware operating-point policy in v2.

## Training diagnostics

SLIC–RF plateaued by approximately 200–300 trees. At 300 trees its fitting
ROC-AUC was 0.999903 and its inner-diagnostic ROC-AUC was 0.911046. Mean mask
IoU was 0.593372 on the sampled fitting images and 0.450435 on inner-diagnostic
images, showing a material held-image generalisation gap.

U-Net v1 improved from an inner-diagnostic pixel-aggregate IoU of 0.360926 at
epoch 1 to a peak of 0.595633 at epoch 21. Early stopping occurred at epoch 27
and restored the best-loss checkpoint. With that checkpoint, mean per-image IoU
was 0.515772 on sampled fitting images and 0.524959 on inner-diagnostic images.
The epoch-to-epoch diagnostic curve was noisy, motivating small-batch-safe
normalisation in v2.

## Reproduction boundary

Prediction reproduction is anchored by the saved weights/model, run
configuration, development manifest, environment snapshot, seed, and Git
revision. A detached worktree at revision `8625b61` can recover the exact
tracked implementation without overwriting current development.

Bit-for-bit retraining is not guaranteed because GPU kernels and multi-worker
loading can be nondeterministic. The saved fitted models are therefore the
authoritative v1 checkpoint for reproducing predictions. The local working
notebooks preserve the executed outputs but remain intentionally untracked.

The executed local U-Net notebook at checkpoint time has SHA-256 digest
`f9cae5e9ecb5c3a614f834abcd142c92f8b4558299c467de022017190c9c1970`.
Its Drive backup should retain that digest so the executed orchestration and
displayed outputs can be distinguished from later notebook revisions.
