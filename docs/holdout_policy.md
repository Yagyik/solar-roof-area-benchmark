# Final holdout policy

The complete official RID2 `test_split_512.csv` is the sealed final holdout.

Individual method notebooks must not load its image names, images, masks, summary
statistics, or examples. All model fitting, threshold selection, prompt design,
post-processing choices, and sensitivity work use only a deterministic
train/validation subdivision of `training_split_512.csv`.

The final comparison notebook will be the only notebook allowed to open the
official test split. It will require frozen configuration files for all three
methods before it can run.

