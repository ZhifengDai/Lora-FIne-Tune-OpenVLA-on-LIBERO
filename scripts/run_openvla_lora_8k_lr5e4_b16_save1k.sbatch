#!/bin/bash
#SBATCH --account=p33200
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=openvla_8k_lr5e4_b16
#SBATCH --output=/projects/p33200/zhifeng/libero_openvla/logs/openvla_lora_8k_lr5e4_b16_%j.out
#SBATCH --error=/projects/p33200/zhifeng/libero_openvla/logs/openvla_lora_8k_lr5e4_b16_%j.err

set -e

module load singularity/default

cd /projects/p33200/zhifeng/libero_openvla
mkdir -p runs adapter_tmp_lr5e4_b16_save1k logs hf_cache tmp

singularity exec --nv --bind /projects:/projects \
  /projects/p33200/zhifeng/libero_openvla/libero_openvla_full.sif \
  bash -lc '
    export PYTHONNOUSERSITE=1
    export HF_HOME=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export HF_HUB_CACHE=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export TRANSFORMERS_CACHE=/projects/p33200/zhifeng/libero_openvla/hf_cache
    export TMPDIR=/projects/p33200/zhifeng/libero_openvla/tmp
    export TF_CPP_MIN_LOG_LEVEL=2
    export WANDB_MODE=offline
    export TOKENIZERS_PARALLELISM=false

    cd /opt/openvla

    torchrun --standalone --nnodes=1 --nproc_per_node=1 vla-scripts/finetune.py \
      --vla_path /projects/p33200/zhifeng/libero_openvla/models/openvla-7b \
      --data_root_dir /projects/p33200/zhifeng/libero_openvla/rlds_datasets/modified_libero_rlds \
      --dataset_name libero_spatial_no_noops \
      --run_root_dir /projects/p33200/zhifeng/libero_openvla/runs \
      --adapter_tmp_dir /projects/p33200/zhifeng/libero_openvla/adapter_tmp_lr5e4_b16_save1k \
      --batch_size 16 \
      --max_steps 8000 \
      --save_steps 1000 \
      --save_latest_checkpoint_only False \
      --learning_rate 5e-4 \
      --grad_accumulation_steps 1 \
      --shuffle_buffer_size 10000 \
      --image_aug True \
      --use_lora True \
      --lora_rank 32 \
      --lora_dropout 0.0 \
      --use_quantization False \
      --wandb_project openvla \
      --wandb_entity none \
      --run_id_note official_rlds_8k_lr5e4_b16_save1k
  '
