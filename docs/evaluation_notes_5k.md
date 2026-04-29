# Evaluation Notes: 5k Checkpoint

## Setup

- Model: OpenVLA-7B
- Fine-tuning method: LoRA
- Dataset: `libero_spatial_no_noops`
- Checkpoint: 5000 steps
- Learning rate: `5e-4`
- Batch size: `16`
- LoRA rank: `32`
- LoRA dropout: `0.0`
- Evaluation suite: LIBERO-Spatial
- Trials per task: `5`
- Total rollouts: `50`
- Rendering backend: OSMesa

## Overall Result

| Metric | Count | Rate |
|---|---:|---:|
| Official success | 6 / 50 | 12% |
| Manual success | 11 / 50 | 22% |

The 5k checkpoint shows clear learning progress compared with earlier failed runs, but the behavior is highly task-dependent. Some tasks exhibit stable successful behavior, while others show almost no motion.

## Task-Level Summary

| Task ID | Manual Success | Summary |
|---:|---:|---|
| 1 | 1 / 5 | Mostly no motion; one successful rollout |
| 2 | 5 / 5 | Stable success across all trials |
| 3 | 0 / 5 | Almost no motion |
| 4 | 1 / 5 | Mostly no motion; one successful rollout |
| 5 | 0 / 5 | Repeated grasp attempts on the bowl inside the cabinet, but failed to grasp |
| 6 | 0 / 5 | Almost no motion |
| 7 | 3 / 5 | Partially stable success |
| 8 | 1 / 5 | Mostly no motion; one successful rollout |
| 9 | 0 / 5 | Almost no motion |
| 10 | 0 / 5 | Almost no motion |

## Interpretation

The 5k checkpoint is not a fully failed policy. It demonstrates partial visual-language-action grounding:

- Task 2 achieved 5/5 manual success.
- Task 7 achieved 3/5 manual success.
- Task 5 repeatedly attempted to grasp the correct target object but failed to complete a stable grasp.

However, the model has not learned a robust general LIBERO-Spatial policy:

- Several tasks showed almost no motion.
- Success was concentrated in a small number of tasks.
- Grasp precision and recovery behavior remained weak.
- The official predicate-based evaluator reported fewer successes than manual inspection.

## Failure Modes

Observed failure modes include:

- `no_motion`: the robot barely moved or did not initiate a meaningful action.
- `mostly_no_motion`: most trials showed little motion, with occasional successful behavior.
- `missed_grasp`: the robot reached toward the target but failed to grasp it.
- `partial_success`: the robot showed meaningful task behavior but was not consistently successful.

## Conclusion

The 5k checkpoint provides evidence that the LoRA fine-tuning configuration is learning useful behavior, but the policy is still unstable and uneven across tasks. The next step is to evaluate later checkpoints, especially 6k, 7k, and 8k, to determine whether additional training improves task coverage and manipulation precision or leads to overfitting/behavior degradation.
