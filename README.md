# Motion Prediction V1: Scene-Level Baseline on nuScenes

Standalone scene-level multi-agent motion prediction baseline built on the nuScenes dataset. The repository is organized around a clean artifact-based pipeline, local scene context in the ego frame, and a compact two-stage anchor-based forecasting model.

## Overview

This project focuses on a practical and readable motion prediction stack rather than a large experimental framework. The core idea is simple:

- build scene-centric training samples offline
- keep the model input contract clean and stable
- represent local map structure explicitly with vectorized geometry
- predict future motion with a staged anchor-based decoder

The result is a repository that is easier to inspect, train, and extend without dragging along older experimental code paths.

## Data Pipeline

The training path is centered on offline artifact generation. Instead of rebuilding the full scene representation during every training step, the repository separates preprocessing from model execution:

- raw nuScenes access and indexing live in `data/`
- offline artifact generation lives in `preprocessing/`
- the final artifact-driven training path lives in `motion_v1/`

This keeps training iterations fast and makes model behavior easier to compare across runs because the batch contract stays fixed once artifacts are created.

## Model Design

The model is built as a scene-level baseline with three main ideas:

- agent motion history is encoded into compact per-agent tokens
- map elements and static scene objects are encoded separately and fused into the same scene representation
- future prediction is produced by a two-stage anchor-based decoder that first captures near-term intent and then rolls that information into the final trajectory forecast

The implementation is intentionally compact: the training entry point is isolated in `train.py`, and the main V1 logic is concentrated in a small number of readable modules.

## Repository Layout

```text
motion_nuscenes/
├── motion_v1/
├── data/
├── preprocessing/
├── docs/
├── .gitignore
├── README.md
└── train.py
```

- `motion_v1/` contains the main V1 dataloader, geometry helpers, category mapping, and model code
- `data/` contains nuScenes-facing dataset and utility code
- `preprocessing/` contains offline preprocessing utilities for artifact generation
- `train.py` is the standalone training entry point

## Qualitative Example

The figure below shows a BEV scene example with motion history, ground-truth future, and predicted future in the ego frame.

![BEV trajectory example](docs/assets/bev_example.png)

## Notes

- the repository is intended to stay lightweight and code-focused
- checkpoints, generated artifacts, and other heavy outputs are ignored by default
- paths are kept portable and are passed through arguments instead of being hardcoded to one machine
