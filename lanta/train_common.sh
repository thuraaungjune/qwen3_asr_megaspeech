# Sourced by the sbatch scripts after they set RUN_NAME and TRAIN_CSVS.
# Any variable below can be overridden at submit time, e.g.
#   sbatch --export=ALL,LR=1e-5,EPOCHS=2 lanta/train_filtered_unrefined.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                       # = training/qwen3_asr_megaspeech (submit from there)
source lanta/config.sh

module load Mamba/23.11.0-0
set +u                                       # conda's activate scripts reference unset variables
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=${SLURM_GPUS_ON_NODE:-4}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}                  # >0 overrides EPOCHS (use for a quick test)
LR=${LR:-2e-5}
BS=${BS:-16}                                # per GPU
ACCUM=${ACCUM:-2}                           # global batch = NGPU * BS * ACCUM = 128 by default
WORKERS=${WORKERS:-12}                      # per GPU; LANTA GPU nodes have 64 cores
SAVE_STEPS=${SAVE_STEPS:-2000}
EVAL_STEPS=${EVAL_STEPS:-2000}
OUT=${OUT:-$CKPT_ROOT/$RUN_NAME}
mkdir -p "$OUT"

echo "run=$RUN_NAME  gpus=$NGPU  out=$OUT"
echo "train: $TRAIN_CSVS"
echo "eval:  $EVAL_CSVS"
nvidia-smi --query-gpu=name,memory.total --format=csv

torchrun --standalone --nproc_per_node "$NGPU" finetune_qwen3_asr.py \
  --model_name_or_path "$BASE_MODEL" \
  --train_csvs $TRAIN_CSVS \
  --eval_csvs $EVAL_CSVS \
  --output_dir "$OUT" \
  --language th \
  --max_audio_seconds 30 \
  --bf16 --tf32 True \
  --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$BS" --per_device_eval_batch_size "$BS" \
  --gradient_accumulation_steps "$ACCUM" \
  --learning_rate "$LR" --lr_scheduler_type cosine --warmup_ratio 0.02 --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataloader_num_workers "$WORKERS" --dataloader_persistent_workers --dataloader_prefetch_factor 4 \
  --logging_steps 50 --logging_dir "$OUT/tb" --report_to tensorboard \
  --eval_strategy steps --eval_steps "$EVAL_STEPS" \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 3 \
  --ddp_find_unused_parameters False \
  --seed 42 \
  "$@"
