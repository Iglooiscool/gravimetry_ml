# Model Packages

The public model APIs are grouped by the number of neural models used in the
reconstruction system.

## `one_model`

`GradientToMaskModel` is the official model. It receives noisy gradient
features and predicts mask logits directly using the coordinate-aware decoder.

## `two_models`

`GradientToCoefficientModel` predicts shape coefficients from gradient
features. `CoefficientToMaskModel` reconstructs a mask from those coefficients.
`TwoModelSystem` documents the connection between the two models.

## `three_models`

The experimental system contains one coefficient model, one general mask model,
and one two-circle specialist mask model. `ThreeModelSystem` documents those
roles. Routing remains experimental and is not the official path.

## Shared Components

`shared` exposes normalization, prediction, seeding, and evaluation helpers.
The lower-level `stage1`, `stage2`, and `task9` packages remain compatibility
components while the model-count APIs become the primary public interface.
