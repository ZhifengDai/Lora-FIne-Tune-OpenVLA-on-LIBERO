
# LoRA Fine-Tuning OpenVLA on LIBERO

This repository documents a reproduction and analysis of LoRA fine-tuning OpenVLA-7B on the LIBERO-Spatial benchmark using the official modified LIBERO RLDS dataset.

## Overview

- Model: OpenVLA-7B

- Dataset: `libero_spatial_no_noops`

- Fine-tuning: LoRA

- Benchmark: LIBERO-Spatial

- Evaluation: rollout success rate and rollout video inspection

- Platform: Northwestern Quest GPU cluster

## Repository Structure

```text

scripts/   Training and evaluation scripts

patches/   Local patches for evaluation and video naming

logs/      Selected training logs

results/   Checkpoint summaries and evaluation tables

docker/    Docker/Singularity environment files

docs/      Evaluation notes and troubleshooting records

## Preliminary Evaluation Result

A preliminary evaluation was conducted on the 5k LoRA checkpoint using LIBERO-Spatial with 5 trials per task.

| Checkpoint | Official Success | Manual Success | Notes |
|---|---:|---:|---|
| 5k | 6 / 50 (12%) | 11 / 50 (22%) | Strong task imbalance; Task 2 achieved 5/5 manual success and Task 7 achieved 3/5. Several tasks showed almost no motion. |

See [`docs/evaluation_notes_5k.md`](docs/evaluation_notes_5k.md) and [`results/task_level_results_5k.csv`](results/task_level_results_5k.csv) for details.

## Training Metrics

The 8k LoRA training run showed stable optimization. The training loss decreased rapidly in the early stage and continued to improve slowly afterward. The L1 action loss also decreased over training, while action accuracy stabilized around 0.35–0.45.

![Training metrics](results/figures/training_metrics_8k_lr5e4_b16.png)

## Notes

Large files such as model weights, checkpoints, RLDS datasets, rollout videos, and Singularity images are not included in this repository.
