# Experiment configurations

Each reportable run will use a versioned configuration that fixes the method,
parameters, model identifiers, random seed, input manifest, and runtime
requirements.

Machine-specific data and Drive locations will live in the ignored
`paths.local.yaml`. They will not be embedded in portable manifests or committed
configuration files.

