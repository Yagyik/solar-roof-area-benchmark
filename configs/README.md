# Experiment configurations

Each reportable run will use a versioned configuration that fixes the method,
parameters, model identifiers, random seed, input manifest, and runtime
requirements.

Machine-specific data and Drive locations will live in the ignored
`paths.local.yaml`. They will not be embedded in portable manifests or committed
configuration files.

Current roof-method configurations are:

- `roof_slic_rf.yaml`: CPU, medium development run by default;
- `roof_unet.yaml`: GPU, smoke training run by default;
- `roof_grounded_sam.yaml`: GPU, smoke prompt-development run by default.

The prompted configuration pins both model repositories to immutable revisions
and records individual checkpoint sizes. Model weights and caches remain in
Google Drive and are excluded from Git.
