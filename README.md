# MFCRAF

Missing-aware Feature Completion and Reliability-Adaptive Fusion for Incomplete Multi-modal Person Re-Identification

This repository provides the core implementation of MFCRAF, a CLIP-based framework for person re-identification with missing modalities. The framework combines modality-aware representation learning, cross-modal feature completion, and reliability-adaptive feature fusion.

## Core Modules

- State-Conditioned Part-Aware LoRA (SPA-LoRA): adapts visual features using spatial parts and modality-availability-conditioned routing.
- Missing-aware Robust Encoding: represents observed and missing modality states.
- Cross-modal Feature Completion: estimates features for missing modalities from available observations.
- Reliability-Adaptive Fusion: predicts modality-specific reliability scores and combines observed and completed features using adaptive weights.

## Repository Structure

- `core/spa_lora.py`: SPA-LoRA implementation.
- `core/missing_aware_encoding.py`: missing-aware feature encoding.
- `core/cross_modal_completion.py`: cross-modal completion module.
- `core/reliability_adaptive_fusion.py`: reliability-based feature fusion.
- `core/mmencoder_withlora.py`, `core/objectives.py`, `core/options.py`: encoder components, objectives, and options.
- `configs/`: example experiment configuration.
- `test.py`: frozen-checkpoint evaluation entry point.

## Setup and Evaluation

The core modules are implemented in Python and PyTorch. Install a compatible PyTorch environment and integrate the modules into a CLIP-based ReID training/evaluation project.

An example evaluation command, after integrating the required project utilities and datasets, is:

```bash
python test.py \
  --config_file configs/RSTPReid_dropout0_rgb03_lora4.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --missing_rates 0.0 0.2 0.3 0.5 \
  --eval_seed 42
```

The evaluation entry point supports missing-rate evaluation and recording of evaluation outputs. Adjust the configuration, dataset paths, and checkpoint to match your setup.

Note: This is a core-code release, not a standalone reproduction package. The full training pipeline, dataset loaders, supporting utilities, datasets, and trained checkpoints are not included. In particular, `test.py` depends on project-level modules that are not part of this repository.

## Code Availability

Source code: https://github.com/Sophia-Robinson-maker/code
