#!/bin/bash
#SBATCH -J qasr06-smoke
#SBATCH -p gpu
#SBATCH -A lt200217
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out

# 50 steps on 20k rows of the filtered set, eval + save at step 25, final CER on 50 rows.
# Checks env, data paths, memory and DDP before you spend days of GPU time.
#   sbatch lanta/smoke_test.sbatch        (try BS=32 with --export=ALL,BS=32 to probe memory)

source "$SLURM_SUBMIT_DIR/lanta/config.sh"
RUN_NAME=smoke-test
TRAIN_CSVS="$FILTERED/gigaspeech-train.csv $FILTERED/scraped.csv $FILTERED/yodas.csv $SHARED_TRAIN"
MAX_STEPS=50 SAVE_STEPS=25 EVAL_STEPS=25 OUT=$CKPT_ROOT/smoke-test-$SLURM_JOB_ID
source "$SLURM_SUBMIT_DIR/lanta/train_common.sh" \
  --max_train_samples 20000 --eval_max_samples 200 --final_cer_samples 50 --logging_steps 5
