---
### Dropout Ablation: LoRA Dropout 0.0 vs 0.1

To evaluate whether LoRA dropout improves OpenVLA fine-tuning stability, I ran an additional 8k-step experiment with `lora_dropout=0.1` while keeping the other settings unchanged: `learning_rate=5e-4`, `batch_size=16`, `lora_rank=32`, and `image_aug=True`.

| Setting | Official Success | Manual Success | Key Observation |
|---|---:|---:|---|
| LoRA dropout = 0.0 | 14 / 80 (17.5%) | 19 / 80 (23.75%) | Some harder-task successes, but strong task imbalance |
| LoRA dropout = 0.1 | 18 / 80 (22.5%) | 22 / 80 (27.5%) | Higher overall success, stronger Task 1/2/4 performance, but still task-dependent |

The dropout=0.1 run improved both official and manually verified success rates. The official success rate increased from **17.5% to 22.5%**, and the manually verified success rate increased from **23.75% to 27.5%**. This suggests that LoRA dropout helped improve average rollout robustness.

### Task-Level Result for LoRA Dropout 0.1

| Task ID | Manual Success | Success Rate | Observation |
|---:|---:|---:|---|
| 1 | 6 / 8 | 75.0% | Strong improvement with multiple successful rollouts |
| 2 | 5 / 8 | 62.5% | Improved after manual inspection; several official failures were visually successful |
| 3 | 1 / 8 | 12.5% | Mostly failed, but no longer completely zero-success |
| 4 | 7 / 8 | 87.5% | Best-performing task under dropout=0.1 |
| 5 | 0 / 8 | 0.0% | Still failed; harder manipulation behavior was not improved |
| 6 | 2 / 8 | 25.0% | Some success, but weaker than the dropout=0.0 checkpoint |
| 7 | 0 / 8 | 0.0% | Persistent failure / no-motion behavior |
| 8 | 0 / 8 | 0.0% | Persistent failure / no-motion behavior |
| 9 | 1 / 8 | 12.5% | One successful rollout |
| 10 | 0 / 8 | 0.0% | Persistent failure / no-motion behavior |

### Interpretation of the Dropout Result

Adding `lora_dropout=0.1` improved the overall success rate compared with the dropout=0.0 baseline. This suggests that dropout helped regularize the LoRA adapter and reduced overfitting to specific visual-action patterns in some tasks.

However, the improvement was not uniform across all LIBERO-Spatial tasks. Task 1, Task 2, and Task 4 became noticeably stronger, while Task 5 and Task 6 became weaker compared with the dropout=0.0 checkpoint. This indicates that dropout changed the task-level behavior distribution rather than simply improving every task.

One possible explanation is that dropout made the policy less dependent on a small set of task-specific visual or action features. This helped reduce conservative or no-motion behavior in some tasks, improving average robustness. At the same time, it may have weakened some precise task-specific manipulation patterns that were useful for harder or previously well-performing tasks.

Tasks 7, 8, and 10 still showed 0/8 manual success, indicating that LoRA dropout alone does not fully solve cross-task generalization or manipulation robustness. Some remaining failures may come from insufficient visual-language grounding, unstable action-token prediction, dataset imbalance, or the difficulty of precise object manipulation in LIBERO.

Overall, `lora_dropout=0.1` improved average success rate, but it did not produce a uniform improvement across all tasks. Future work should evaluate intermediate checkpoints, test smaller dropout values such as `0.05`, and separately analyze no-motion failures versus attempt-but-fail behaviors.

## Demos and Qualitative Observations

### Representative rollout GIFs from 8k checkpoint

| Task 5 success | Task 9 shaking success |
|---|---|
| ![Task 5 success](results/videos/8k_examples/task05_success.gif) | ![Task 9 shaking success](results/videos/8k_examples/task09_shaking_success.gif) |
| Task 5: a harder manipulation case that succeeds at 8k | Task 9: unstable shaking behavior followed by successful completion |

| Task 2 placement failure | Task 2 grasp/pick failure |
|---|---|
| ![Task 2 put failure](results/videos/8k_examples/task02_put_failed_fail.gif) | ![Task 2 pick failure](results/videos/8k_examples/task02_pick_failed_fail.gif) |
| Task 2: object moved toward the goal but failed final placement | Task 2: failed grasp or pickup despite task attempt |

### Representative rollout videos from 5k checkpoint

| Stable success | Partial placement failure |
|---|---|
| ![Task 2 success](results/videos/5k_examples/task02_success.gif) | ![Task 1 partial placement failure](results/videos/5k_examples/task01_partial_place_fail.gif) |
| Task 2: successful rollout | Task 1: object is partially placed but not fully inside the target region |

| Repeated grasp attempt | No-motion failure |
|---|---|
| ![Task 5 repeated grasp fail](results/videos/5k_examples/task05_repeated_grasp_fail.gif) | ![Task 6 no motion fail](results/videos/5k_examples/task06_no_motion_fail.gif) |
| Task 5: repeated grasp attempts but failed to secure the bowl | Task 6: little or no meaningful motion |

Evaluation was performed on the 5k LoRA checkpoint using LIBERO-Spatial with 5 trials per task.

Several rollouts showed meaningful learned behavior, while others showed little or no motion. Manual video inspection was used in addition to the official simulator success metric because some visually successful rollouts were marked as failures by the strict automatic predicate evaluator.


The most important qualitative finding is that the 5k checkpoint is not a fully failed policy. It shows partial vision-language-action grounding on some tasks, especially Task 2 and Task 7. However, the behavior is highly task-dependent and not yet a robust general LIBERO-Spatial policy.

---

## Training Metrics

The 8k LoRA training run showed stable optimization. The training loss decreased rapidly in the early stage and continued to improve slowly afterward. The L1 action loss also decreased over training, while action accuracy stabilized around 0.35–0.45.

### Train Loss

<table>
<tr>
<td width="55%">
<img src="results/figures/train_loss_8k_lr5e4_b16.png" width="100%">
</td>
<td width="45%">

The training loss drops sharply at the beginning and then decreases more gradually. This indicates that the model quickly adapts to the LIBERO-Spatial data distribution in the early stage, while later training mainly brings slower incremental improvement.

The 5k checkpoint corresponds to the middle-late stage of this curve, while the 8k checkpoint corresponds to the end of training.

</td>
</tr>
</table>

### L1 Action Loss

<table>
<tr>
<td width="55%">
<img src="results/figures/l1_loss_8k_lr5e4_b16.png" width="100%">
</td>
<td width="45%">

The L1 action loss shows an overall decreasing trend, suggesting that the predicted continuous action values become closer to the ground-truth actions during LoRA fine-tuning.

There is a temporary spike around the middle of training, but the curve later recovers and continues decreasing, indicating that the training process remains stable overall.

</td>
</tr>
</table>

### Action Accuracy

<table>
<tr>
<td width="55%">
<img src="results/figures/action_accuracy_8k_lr5e4_b16.png" width="100%">
</td>
<td width="45%">

The action-token accuracy quickly rises in the early stage and then fluctuates around approximately 0.35–0.45.

This suggests that the model learns useful action-token patterns, but the prediction remains noisy. The gap between improved training metrics and limited rollout success also shows that lower loss does not directly guarantee robust robot manipulation performance.

</td>
</tr>
</table>

---

## Evaluation Results

### Overall checkpoint comparison

| Checkpoint | Trials per Task | Total Rollouts | Official Success | Manual Success | Key Observation |
|---|---:|---:|---:|---:|---|
| 5k | 5 | 50 | 6 / 50 (12.0%) | 11 / 50 (22.0%) | More exploratory behavior; Task 2 was strong, but many tasks remained unstable |
| 8k | 8 | 80 | 14 / 80 (17.5%) | 19 / 80 (23.75%) | Higher official success rate, stronger behavior on some tasks, but still highly task-dependent |

The 8k checkpoint improved the official success rate from **12.0% to 17.5%** and slightly improved the manually verified success rate from **22.0% to 23.75%**. However, the improvement was not uniform across tasks. Some tasks became stronger, while others still showed almost no meaningful motion.

### 8k task-level manual evaluation

| Task ID | Manual Success | Success Rate | Observation |
|---:|---:|---:|---|
| 1 | 3 / 8 | 37.5% | Some successful rollouts, but behavior remains unstable |
| 2 | 2 / 8 | 25.0% | Lower than the 5k checkpoint; more failures appeared |
| 3 | 0 / 8 | 0.0% | All eight rollouts showed little or no meaningful motion |
| 4 | 2 / 8 | 25.0% | Occasional success |
| 5 | 3 / 8 | 37.5% | Harder task, but successful cases appeared at 8k |
| 6 | 7 / 8 | 87.5% | Best-performing task at 8k |
| 7 | 0 / 8 | 0.0% | Mostly failed or no-motion behavior |
| 8 | 1 / 8 | 12.5% | Occasional success |
| 9 | 1 / 8 | 12.5% | One success after unstable shaking behavior |
| 10 | 0 / 8 | 0.0% | Mostly failed or no-motion behavior |

### Interpretation

The 8k checkpoint shows a modest improvement over the 5k checkpoint in terms of official success rate, but the qualitative behavior remains uneven.

Several important patterns were observed:

- **Improved task-specific behavior:** Task 6 achieved 7/8 manual success, and Task 5, a harder manipulation task, showed successful cases at 8k.
- **More attempts before failure:** Compared with earlier checkpoints, more failed rollouts showed meaningful attempts, such as repeated grasping or shaking before failure.
- **Persistent no-motion failures:** Some tasks, especially Task 3, Task 7, and Task 10, still showed little or no meaningful motion.
- **Strong task imbalance:** The model did not improve uniformly across all LIBERO-Spatial tasks. Success was concentrated in a subset of tasks.

Overall, the 8k checkpoint suggests that longer LoRA fine-tuning improved partial task grounding and some manipulation behaviors, but it did not fully solve manipulation robustness or cross-task generalization.

### Interpretation Based on Checkpoint Comparison

The 8k checkpoint improves the official success rate over the 5k checkpoint, increasing from 6/50 (12.0%) to 14/80 (17.5%). Manual inspection also shows a slight improvement from 11/50 (22.0%) to 19/80 (23.75%). This suggests that longer LoRA fine-tuning improves some aspects of action prediction and task execution.

However, the improvement is not uniform across tasks. The 8k checkpoint performs strongly on some tasks, such as Task 6 with 7/8 manual success, and also produces successful cases on the harder Task 5. At the same time, Tasks 3, 7, and 10 still show little or no meaningful motion. This indicates strong task imbalance rather than general improvement across all LIBERO-Spatial tasks.

A likely explanation is that the 5k checkpoint is less converged and more exploratory: it often attempts to move or grasp, but lacks manipulation precision. The 8k checkpoint is more converged and achieves lower training loss, but may become more conservative or task-biased under some visual-language states. Since the LoRA dropout was set to 0.0, overfitting to specific visual states may also contribute to no-motion or action-collapse behavior.

In addition, the gap between official success and manual success suggests that the LIBERO/robosuite predicate-based evaluator can be stricter than visual inspection. Some rollouts that appear visually successful may fail due to small position errors, stability checks, or timeout constraints.

Overall, the 8k checkpoint provides modest quantitative improvement but does not fully solve robust manipulation. The next step is to evaluate 6k and 7k checkpoints to determine whether an intermediate checkpoint better balances exploration, grasp precision, and task coverage.

---

## Project Overview

This project fine-tunes OpenVLA-7B on the LIBERO-Spatial benchmark using LoRA.

- **Base model:** OpenVLA-7B
- **Benchmark:** LIBERO-Spatial
- **Dataset:** `libero_spatial_no_noops`
- **Fine-tuning method:** LoRA
- **LoRA rank:** 32
- **Batch size:** 16
- **Learning rate:** 5e-4
- **Checkpoint saving:** every 1000 steps
- **Evaluation:** rollout success rate + manual video inspection
- **Platform:** Northwestern Quest GPU cluster

The main experiment saves independent checkpoints from 1k to 8k steps. The current evaluation focuses on the 5k checkpoint, with later checkpoints planned for comparison.

---

## Environment Setup

The experiments were run on the Northwestern Quest GPU cluster using Singularity.

Key components:

| Component | Setting |
|---|---|
| Python | 3.10 |
| Model framework | PyTorch / Transformers |
| Robot simulation | MuJoCo, robosuite, LIBERO |
| Dataset format | RLDS / TensorFlow |
| Container | Singularity image |
| Rendering backend | OSMesa fallback |
| GPU acceleration | CUDA for model training and inference |

Large files such as model weights, RLDS datasets, checkpoints, rollout videos, and Singularity images are not included in this repository.

---

## Current Status

- Completed the original 8k LoRA fine-tuning run with `lora_dropout=0.0`.
- Completed an additional 8k LoRA dropout ablation run with `lora_dropout=0.1`.
- Evaluated 5k and 8k checkpoints using LIBERO rollout generation and manual video inspection.
- The original dropout=0.0 8k checkpoint achieved 14/80 official success and 19/80 manual success.
- The dropout=0.1 8k checkpoint improved to 18/80 official success and 22/80 manual success.
- Representative 5k and 8k rollout GIFs were added to the README.
- Next step: evaluate intermediate checkpoints and test smaller dropout values such as 0.05.
