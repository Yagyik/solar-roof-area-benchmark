# Grounded-SAM roof v2 medium-run checkpoint

Checkpoint date: 2026-08-21

Run ID: `roof_grounded_sam_v2_medium_20260821T113645Z`

The run used 40 prompt-development images and 100 outer-validation images. It
accessed zero official final-test rows. The self-contained run directory and
downloadable ZIP are stored under
`/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/`.

## Frozen deployable route

- Roof prompts: `roof of a building`, `roof attached to a building`, and
  `building rooftop`.
- Strategy: un-clipped hierarchical building→roof detection followed by SAM.
- Parent and roof box thresholds: 0.25, varied together in v2.
- Parent and per-crop roof detection caps: 20. The v2 un-clipped hierarchy did
  not impose an additional global post-deduplication cap; v3 makes that global
  limit explicit while retaining the requested value of 20.

## Outer-validation performance

- Mean IoU: 0.487649
- Median IoU: 0.573172
- Mean absolute area error: 175.099072 m²
- Mean precision: 0.677793
- Mean recall: 0.645135

Grounded-SAM v2 had the strongest mean and median overlap of the medium
candidates, but its area error remained worse than SLIC–RF (110.190464 m²) and
the area-balanced U-Net v1 operating point (117.879680 m²).

## Stage-isolation result

Oracle component boxes followed by the same SAM model achieved mean IoU
0.857559, median IoU 0.898713, and mean absolute area error 28.228480 m² on the
40 prompt-development images. This large gap identifies DINO localization as
the main prompted-pipeline bottleneck.

The v2 threshold sweep moved parent and roof thresholds together. Threshold
0.20 had higher development IoU than 0.25, whereas 0.25 had lower area error;
thresholds at and above 0.35 largely eliminated detections. V3 therefore
separates and diagnoses the two gates rather than adding distractor penalties.

The effective v2 configuration is retained in the Drive bundle and the initial
version-controlled configuration is frozen at
`configs/archive/roof_grounded_sam_v2.yaml`.
