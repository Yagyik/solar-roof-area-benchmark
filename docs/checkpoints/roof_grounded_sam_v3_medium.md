# Grounded-SAM roof v3 medium-run checkpoint

Checkpoint date: 2026-08-21

Run ID: `roof_grounded_sam_v3_medium_20260821T125806Z`

The run used 40 prompt-development images and 100 outer-validation images. It
accessed zero official final-test rows. The self-contained run directory and
downloadable ZIP are stored under
`/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/`.

## Frozen deployable route

- Relational roof prompts and the un-clipped hierarchy inherited from v2.
- Independently selected parent threshold: 0.15.
- Independently selected roof threshold: 0.30.
- Final parent and roof-box limits: 20.
- No distractor prompts or penalties.

## Outer-validation performance

- Mean IoU: 0.522447
- Median IoU: 0.587211
- Mean absolute area error: 142.564224 m²
- Mean precision: 0.671177
- Mean recall: 0.730428

Relative to v2, mean IoU improved by 0.034798, recall improved by 0.085293,
and mean absolute area error fell by 18.6%. V3 is the strongest overlap method
at this checkpoint, although SLIC–RF and the area-balanced U-Net v1 operating
point retain lower area error.

## Localization diagnosis

At the selected parent threshold, prompt-development roof-pixel coverage was
0.995919 and oracle-component coverage was 0.936709. The parent cap was reached
on half the development images, with a mean of 14.875 parent boxes and 69.0% of
the image admitted as candidate area.

Outer roof-box precision/recall at IoU 0.50 were 0.294032 and 0.506266. When a
roof was matched, its mean matched-box IoU was 0.688754, but the mean best IoU
over all oracle components was only 0.516888. The remaining bottleneck is
therefore child roof localization plus apparent redundancy in parent boxes.

V4 preserves prompts, models, thresholds grids, SAM, and final caps while
adding class-agnostic NMS to parent proposals before the final 20-box cap.
