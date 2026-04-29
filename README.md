
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
