# Project working rules

## Execution boundary

- Develop and review code in VS Code, but execute project Python, notebooks,
  benchmarks, and tests only on a Google Colab runtime unless the user
  explicitly changes this rule.
- Each executable experiment must declare its required accelerator and minimum
  resources, inspect the actual Colab runtime at startup, and fail early when
  the runtime does not satisfy those requirements.
- TPU is not a substitute for GPU unless a stage is explicitly implemented and
  validated for TPU execution.

## Incremental workflow

- Work on one user-approved checkpoint at a time and explain its contents before
  starting the next checkpoint.
- Keep notebooks focused on exploration, orchestration, visual diagnostics, and
  reporting. Put reusable logic under `src/roof_panel_placement/`.
- Do not refactor all prototype notebook code at once. Extract only the code
  required by the current experimental stage.

## Notebook and Git policy

- Preserve working notebooks locally and in Google Drive during experimentation.
- Do not commit experimental notebooks. Curated final notebooks belong in
  `notebooks/final/` and are committed only after explicit review.
- Never commit RID2 data, archives, model weights, caches, secrets, or bulk run
  artifacts.

## Evaluation policy

- Oracle masks are valid for ground truth, upper bounds, and stage-isolation
  experiments. They must not be inputs to the final deployable pipeline or to
  representative images that describe that pipeline.
- Use fixed development, validation, and test manifests. Do not tune prompts,
  thresholds, or post-processing against the final test cases.
- The classical baseline must be fully classical: no neural networks,
  pretrained representations, or model-generated proposals.
- Save method configuration, Git revision, environment details, per-case
  metrics, aggregate metrics, diagnostics, and representative figures for every
  reportable run.

## Artifact policy

- A reportable run must produce a self-contained downloadable bundle in Google
  Drive. It must include machine-readable metrics and configuration as well as
  human-readable diagnostics.
- The main report presents one selected best deployable method. Method and
  parameter sensitivity analyses belong in appendix-ready outputs.

