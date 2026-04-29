#!/bin/bash
#SBATCH --account=p33200
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_5k_10trials
#SBATCH --output=/projects/p33200/zhifeng/libero_openvla/logs/eval_5k_10trials_%j.out
#SBATCH --error=/projects/p33200/zhifeng/libero_openvla/logs/eval_5k_10trials_%j.err

set -e
module load singularity/default

cd /projects/p33200/zhifeng/libero_openvla

singularity exec --nv \
  --bind /projects:/projects \
  /projects/p33200/zhifeng/libero_openvla/libero_openvla_full.sif \
  bash -lc '
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/projects/p33200/zhifeng/libero_openvla/openvla_eval_work:$PYTHONPATH

    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    export TOKENIZERS_PARALLELISM=false

    export EVAL_TRIALS_PER_TASK=10

    export HF_HOME=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export HF_HUB_CACHE=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export TRANSFORMERS_CACHE=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export TF_CPP_MIN_LOG_LEVEL=2

    cd /projects/p33200/zhifeng/libero_openvla/openvla_eval_work

    echo "=== using openvla_utils ==="
    python3 -c "import experiments.robot.openvla_utils as u; print(u.__file__)"

    echo "=== attention setting ==="
    grep -n "attn_implementation" experiments/robot/openvla_utils.py

    echo "=== video naming setting ==="
    grep -n "task_id\|trial_id\|mp4_path" experiments/robot/libero/libero_utils.py

    echo "=== rendering backend ==="
    echo MUJOCO_GL=$MUJOCO_GL
    echo PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM
    echo EVAL_TRIALS_PER_TASK=$EVAL_TRIALS_PER_TASK

    python3 experiments/robot/libero/run_libero_eval.py \
      --model_family openvla \
      --pretrained_checkpoint /projects/p33200/zhifeng/libero_openvla/runs/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--official_rlds_8k_lr5e4_b16_save1k--image_aug--5000_chkpt \
      --task_suite_name libero_spatial \
      --center_crop True \
      --num_trials_per_task 10 \
      --num_steps_wait 10 \
      --local_log_dir /projects/p33200/zhifeng/libero_openvla/eval_logs \
      --use_wandb False \
      --seed 7 \
      --run_id_note eval_5k_lr5e4_b16_10trials_osmesa
  '

DATE_DIR=$(ls -td /projects/p33200/zhifeng/libero_openvla/openvla_eval_work/rollouts/* 2>/dev/null | head -1 || true)

if [ -n "$DATE_DIR" ]; then
    DEST=/projects/p33200/zhifeng/libero_openvla/eval_videos/5k_lr5e4_b16_10trials_${SLURM_JOB_ID}
    mkdir -p "$DEST"
    cp -v "$DATE_DIR"/*.mp4 "$DEST"/ 2>/dev/null || true
    echo "Copied videos to: $DEST"
fi
