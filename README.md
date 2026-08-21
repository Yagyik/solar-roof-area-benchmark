# Solar Roof Area Benchmark

This project compares three approaches to binary segmentation of all visible
roof pixels in RID2 aerial-image tiles:

1. multiscale SLIC superpixels with hand-crafted features and a Random Forest;
2. a task-trained U-Net; and
3. a prompted detector-and-segmentation pipeline.

Development happens in VS Code and execution happens only in Google Colab.
Working notebooks and large artifacts remain outside Git; reviewed final
notebooks and small result summaries will be added after the methods are frozen.

## Current checkpoint

The shared data, development-split, metrics, runtime, and Drive-artifact modules
are present. Working experiment notebooks are maintained locally and in
Drive, but remain ignored by Git until they are curated:

- `notebooks/working/10_roof_slic_rf_experiments.ipynb`
- `notebooks/working/20_roof_unet_experiments.ipynb`
- `notebooks/working/21_roof_unet_v2_experiments.ipynb`
- `notebooks/working/30_roof_grounded_sam_experiments.ipynb`
- `notebooks/working/31_roof_grounded_sam_v2_experiments.ipynb`
- `notebooks/working/32_roof_grounded_sam_v3_box_diagnostics.ipynb`

Reusable implementations live in `src/roof_panel_placement/segmentation/`.
SLIC–RF runs on CPU; U-Net and Grounding DINO + SAM 2.1 require a GPU runtime.

The official RID2 `test_split_512.csv` is sealed until the final comparison.
See `docs/holdout_policy.md`.

The first medium-run comparison is frozen in
`docs/checkpoints/roof_segmentation_v1_medium.md`; deferred comparison work is
tracked in `docs/experiment_todos.md`. The U-Net v2 medium negative result is
frozen separately in `docs/checkpoints/roof_unet_v2_medium.md`; the first
Grounded-SAM smoke result is frozen in
`docs/checkpoints/roof_grounded_sam_v1_smoke.md`; its v2 medium successor is
frozen in `docs/checkpoints/roof_grounded_sam_v2_medium.md`.
