# Changelog

## ResUNet + clDice update

This update turns the original PlusResUNet boundary-segmentation prototype into a modular, topology-aware trajectory-line segmentation pipeline.

### Added

- Differentiable soft skeletonization and clDice loss.
- Default objective: `0.3 BCE + 0.3 Dice + 0.4 clDice`.
- Hard clDice and a fuller per-image metric suite for evaluation with masks.
- Balanced per-image 512×512 patch sampling.
- Modular `dataset`, `model`, `losses`, `metrics`, and `postprocess` components.
- A synthetic end-to-end smoke test.
- Complete visualization outputs for the `1-negative` and `64` examples.
- Perimeter measurements for detected closed structures.

### Changed

- Input changed from RGB to percentile-normalized grayscale.
- Mixed 256/512-pixel training changed to fixed 512-pixel patches.
- Five-fold cross-validation changed to a reproducible image-level train/validation split.
- Best-checkpoint selection now monitors validation Dice.
- The default binary threshold changed from 0.5 to 0.2.

### Removed from the current tree

- The original monolithic training and prediction scripts.
- The original checkpoint and example outputs, which remain recoverable from commit `5489b56`.
