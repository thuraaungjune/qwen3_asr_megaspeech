#!/bin/bash
#SBATCH -J qasr06-unfilt-unref-lora
#SBATCH -p gpu
#SBATCH -A lt200217
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -t 120:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out

# Model 1 (LoRA r=64 on the LLM + trained projector, merged at the end): Unfiltered audios + Unrefined texts (FULL ORIGINAL)
#   gigaspeech-train, scraped, yodas    <- unfiltered+unrefined
#   ai-cooking, thai-central            <- filtered+unrefined (never filtered, only stored there)
#   eval: gigaspeech-dev                <- filtered+unrefined
# Submit from the repo root:   sbatch lanta/train_unfiltered_unrefined_lora.sbatch
# Resubmitting the same script resumes from the newest checkpoint.

source "$SLURM_SUBMIT_DIR/lanta/config.sh"
RUN_NAME=qwen3-asr-0.6b-megaspeech-unfiltered-unrefined-lora
LORA=1
TRAIN_CSVS="$UNFILTERED/gigaspeech-train.csv $UNFILTERED/scraped.csv $UNFILTERED/yodas.csv $SHARED_TRAIN"
source "$SLURM_SUBMIT_DIR/lanta/train_common.sh"
