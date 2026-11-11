#!/bin/bash
#SBATCH -J qasr06-filt-unref-lora
#SBATCH -p gpu
#SBATCH -A lt200217
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -t 120:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out

# Model 2 (LoRA r=64 on the LLM + trained projector, merged at the end): Filtered audios + Unrefined texts
#   gigaspeech-train, scraped, yodas, ai-cooking, thai-central   <- filtered+unrefined
#   eval: gigaspeech-dev                                          <- filtered+unrefined
# Submit from the repo root:   sbatch lanta/train_filtered_unrefined_lora.sbatch
# Resubmitting the same script resumes from the newest checkpoint.

source "$SLURM_SUBMIT_DIR/lanta/config.sh"
RUN_NAME=qwen3-asr-0.6b-megaspeech-filtered-unrefined-lora
LORA=1
TRAIN_CSVS="$FILTERED/gigaspeech-train.csv $FILTERED/scraped.csv $FILTERED/yodas.csv $SHARED_TRAIN"
source "$SLURM_SUBMIT_DIR/lanta/train_common.sh"
