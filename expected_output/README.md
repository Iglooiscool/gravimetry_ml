# Expected Outputs

This directory contains the presentation-ready artifacts produced by the
official model notebooks. The full raw runs remain under `output/*/runs/` and
are not copied here.

Each stage is grouped by artifact type:

- `fixed_reconstructions/`: fixed benchmark shapes and model reconstructions.
- `summary_json/`: one summary for each training-noise condition.
- `models/`: saved model weights for each training-noise condition.

The fixed benchmark inputs are clean. The sigma shown in each filename is the
training gradient-noise sigma used for the model that produced that artifact.
The official test result is the clean test result. Test-noise robustness
measurements are not included in this directory.
