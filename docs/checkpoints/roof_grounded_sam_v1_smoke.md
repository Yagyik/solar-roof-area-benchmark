# Grounded-SAM roof v1 smoke-run checkpoint

Checkpoint date: 2026-08-21

Run ID: `roof_grounded_sam_smoke_20260821T104500Z`

This was an engineering-scale prompt experiment: six prompt-development cases
and twelve outer-validation cases, with zero official final-test rows accessed.
The downloadable run bundle is stored under
`/content/drive/MyDrive/roof_poc/experiment_artifacts/roof/`.

## Development observations

The original hierarchical building→roof strategy ranked highest on the six
prompt-development images:

| Strategy | Mean IoU | Mean precision | Mean recall | Mean boxes |
|---|---:|---:|---:|---:|
| Hierarchical | 0.415861 | 0.692191 | 0.611351 | 4.333 |
| Independent union | 0.377886 | 0.463186 | 0.795683 | — |
| Joint | 0.342608 | — | — | — |
| Independent consensus | 0.341552 | — | — | — |

The v1 rule selected box threshold 0.20 by mean IoU. Its development mean IoU
was 0.422857 and mean absolute area error was 238.459733 m². Threshold 0.35 had
lower mean IoU (0.391778) but materially lower area error (134.343467 m²), which
motivated the area-aware threshold rule in v2.

## Outer-validation result

With the hierarchy and threshold 0.20 frozen, the twelve outer cases produced:

- mean IoU: 0.509490;
- median IoU: 0.546418;
- mean absolute area error: 290.417600 m².

The masks exhibited large false-positive areas and two zero-IoU failures.
High SAM scores on poor cases indicated that text-conditioned box localization,
rather than SAM confidence alone, was an important bottleneck.

This smoke result is not a final method ranking. Its configuration is frozen at
`configs/archive/roof_grounded_sam_v1.yaml`; v2 is a separate trial.
