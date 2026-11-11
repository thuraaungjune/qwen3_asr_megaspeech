# Shared settings, sourced by setup_env.sh and the sbatch scripts. Edit here, not in each script.

PROJECT=/project/lt200217-thwhis
DATA_ROOT=$PROJECT/data/annotated_audios/original_of_megaspeech_v3
FILTERED=$DATA_ROOT/filtered+unrefined
UNFILTERED=$DATA_ROOT/unfiltered+unrefined

WORK=$PROJECT/$USER/qwen3_asr_megaspeech        # env, HF cache, checkpoints and logs all live here (not $HOME)
ENV_PREFIX=$WORK/envs/qwen3asr
export HF_HOME=$WORK/hf_cache
CKPT_ROOT=$WORK/checkpoints

BASE_MODEL=Qwen/Qwen3-ASR-0.6B-hf

# ai-cooking, thai-central and gigaspeech-dev were never filtered (manually created, trusted),
# so they exist only in filtered+unrefined and are shared by both runs.
SHARED_TRAIN="$FILTERED/ai-cooking.csv $FILTERED/thai-central.csv"
EVAL_CSVS="$FILTERED/gigaspeech-dev.csv"
