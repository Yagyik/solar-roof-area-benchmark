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
are present. The first working notebook is
`notebooks/working/10_roof_slic_rf_experiments.ipynb`.

The official RID2 `test_split_512.csv` is sealed until the final comparison.
See `docs/holdout_policy.md`.

