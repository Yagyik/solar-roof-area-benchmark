# Grounded-SAM roof v4 medium-run checkpoint

Checkpoint date: 2026-08-22

Run ID: `roof_grounded_sam_v4_medium_20260821T162957Z`

The run used 40 prompt-development images and 100 outer-validation images. It
accessed zero official final-test rows. Its self-contained directory and ZIP
are stored under
`/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/`.

## Frozen selected route

- Parent threshold: 0.10.
- Class-agnostic parent NMS threshold: 0.30.
- Roof threshold: 0.30.
- Temporary parent pool: 60 proposals.
- Final parent and roof-box caps: 20 each.
- No distractor prompts or penalties.

## Outer-validation performance

- Mean IoU: 0.496147
- Median IoU: 0.531028
- Mean absolute area error: 133.849856 m²
- Mean precision: 0.620351
- Mean recall: 0.706604
- Roof-box precision/recall at IoU 0.50: 0.230108 / 0.536341

Relative to v3, v4 reduced mean absolute area error by 8.714368 m² but lowered
mean IoU by 0.026300, median IoU by 0.056182, precision by 0.050827, and recall
by 0.023824. It therefore does not replace v3 as the strongest prompted roof
segmentation method.

## Parent-NMS diagnosis

On prompt-development images, the selected configuration suppressed 69.8% of
raw parents and reduced final-cap saturation to 15%, while retaining 99.91% of
roof pixels and 96.20% of oracle components. Nevertheless, candidate parent
regions occupied 73.3% of each image and the raw 60-proposal pool saturated on
42.5% of cases.

The downstream decline indicates that parent inclusion coverage is not a
sufficient crop-quality measure. Aggressive class-agnostic NMS can retain broad
parent boxes while suppressing tighter overlapping boxes; the broad crop still
covers the oracle roof but provides a poorer child-detection context. V4 is
retained as an appendix-ready sensitivity result.
