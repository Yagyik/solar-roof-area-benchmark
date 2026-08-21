# Experiment to-dos

This file records analyses that are deliberately deferred so they are not lost
while individual methods are still being developed.

## Roof segmentation comparison

- [ ] **Paired per-image method comparison.** Once the SLIC–RF, U-Net, and
  prompted-pipeline candidates are frozen, join their per-image results by
  `sample_id`. Report paired mean and median differences, method win rates, and
  roof-area-stratified paired bootstrap confidence intervals for IoU and
  absolute area error. Use development/validation data for method selection and
  repeat the locked comparison exactly once on the untouched final test set.
- [ ] **Roof-area-stratum-specific evaluation.** Report mask IoU, precision,
  recall, signed area error, and absolute area error separately for every fixed
  roof-fraction stratum. Include the case count in each stratum and inspect
  catastrophic failures rather than relying only on an overall mean. Exact
  no-roof performance is not a current project priority.

## Evaluation safeguards

- [ ] Define a practically meaningful improvement threshold alongside
  statistical uncertainty; a narrow confidence interval around a negligible
  improvement should not determine the selected method.
- [ ] Preserve pairing during resampling and retain the fixed roof-area-stratum
  composition when bootstrapping.
- [ ] Freeze model weights, prompts, thresholds, and post-processing before the
  final-test comparison.

