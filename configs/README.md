# Experiment configurations

Each reportable run will use a versioned configuration that fixes the method,
parameters, model identifiers, random seed, input manifest, and runtime
requirements.

Machine-specific data and Drive locations will live in the ignored
`paths.local.yaml`. They will not be embedded in portable manifests or committed
configuration files.

Current roof-method configurations are:

- `roof_slic_rf.yaml`: CPU, medium development run by default;
- `roof_unet.yaml`: preserved v1 GPU experiment;
- `roof_unet_v2.yaml`: GroupNorm and area-aware v2 GPU experiment;
- `roof_grounded_sam.yaml`: preserved v1 GPU prompted experiment;
- `roof_grounded_sam_v2.yaml`: augmented-prompt, parent-box, clipped-hierarchy,
  area-aware v2 GPU experiment;
- `roof_grounded_sam_v3.yaml`: fixed relational hierarchy with independently
  diagnosed parent and roof detector thresholds;
- `roof_grounded_sam_v4.yaml`: class-agnostic parent NMS before the fixed
  20-parent cap, followed by a separately retuned roof gate.
- `roof_method_checkpoint.yaml`: frozen source-run IDs and reporting cohorts
  for the four-method medium-run checkpoint and SLIC walkthrough.

Frozen historical configurations live under `configs/archive/` and must not be
edited when a new trial is introduced.

The prompted configuration pins both model repositories to immutable revisions
and records individual checkpoint sizes. Model weights and caches remain in
Google Drive and are excluded from Git.
