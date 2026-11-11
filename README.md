# Qwen3-ASR-0.6B fine-tuning on MegaSpeech v3 (LANTA)

Full supervised fine-tuning of `Qwen/Qwen3-ASR-0.6B-hf` with the Hugging Face `Trainer`. It produces two models:

| Run | sbatch | Training CSVs |
|---|---|---|
| **Unfiltered audios + unrefined texts** (full original) | `lanta/train_unfiltered_unrefined.sbatch` | `unfiltered+unrefined/{gigaspeech-train, scraped, yodas}` + `filtered+unrefined/{ai-cooking, thai-central}` |
| **Filtered audios + unrefined texts** | `lanta/train_filtered_unrefined.sbatch` | `filtered+unrefined/{gigaspeech-train, scraped, yodas, ai-cooking, thai-central}` |

`ai-cooking`, `thai-central` and `gigaspeech-dev` were never filtered and are stored only in `filtered+unrefined`, so both runs share them.
`gigaspeech-dev` is held out for evaluation, not training. Both runs use the same eval set, so their numbers are comparable.

```
finetune_qwen3_asr.py        training script (any HF TrainingArguments flag works)
inspect_csvs.py              checks the CSV columns, audio paths and durations
lanta/config.sh              paths shared by every script (data, env, HF cache, checkpoints)
lanta/setup_env.sh           one-time: conda env + model download (run on the frontend)
lanta/activate.sh            source it to get the env in your shell
lanta/train_common.sh        torchrun command + hyper-parameters (sourced by the sbatch files)
lanta/smoke_test.sbatch      50-step check, 1 h
lanta/train_*.sbatch         the two real runs
```

## Steps

```bash
ssh <user>@lanta.nstda.or.th
cd /project/lt200217-thwhis/$USER        # or wherever you keep the repo
git clone https://github.com/thuraaungjune/qwen3_asr_megaspeech.git && cd qwen3_asr_megaspeech
mkdir -p logs                            # sbatch writes logs/ here; the job fails if it is missing

# 1. Environment + base model (frontend: compute nodes have no internet)
bash lanta/setup_env.sh

# 2. Check the CSVs: column names, whether the audio paths resolve, total hours
source lanta/activate.sh                 # should print .../envs/qwen3asr/bin/python (Python 3.11)
python inspect_csvs.py $FILTERED/*.csv $UNFILTERED/*.csv

# 3. Smoke test (50 steps, 4 GPUs, ~15 min). Read logs/qasr06-smoke-<jobid>.out
sbatch lanta/smoke_test.sbatch

# 4. The two models
sbatch lanta/train_unfiltered_unrefined.sbatch
sbatch lanta/train_filtered_unrefined.sbatch
```

Monitor runs with `squeue -u $USER` and `tail -f logs/<name>-<jobid>.out`, or with TensorBoard: `tensorboard --logdir $CKPT_ROOT/<run>`.

**Walltime / resuming.** Each job asks for 120 h. If a run doesn't finish in time, submit the same sbatch again and it resumes from the newest `checkpoint-*` in the output folder (checkpoints every 2000 steps, newest 3 kept).

**Outputs** go to `/project/lt200217-thwhis/$USER/qwen3_asr_megaspeech/checkpoints/<run>/`:
- `final/` holds the model and processor, loadable with `Qwen3ASRForConditionalGeneration.from_pretrained` (works with the ablation notebook too);
- `final_eval_predictions.csv` has greedy outputs on 500 gigaspeech-dev rows; the CER is printed at the end of the log.

Push to the Hub from the frontend: `hf upload <user>/<repo> $CKPT_ROOT/<run>/final .`

## Inspect the CSVs first

I haven't seen the CSVs, so the script **guesses the column names**:

| Field | Columns it tries |
|---|---|
| audio | `audio_path`, `audio_filepath`, `path`, `filepath`, `file_path`, `audio`, `wav`, … |
| text | `text`, `transcription`, `transcript`, `sentence`, … |
| duration | `duration`, `duration_s`, … (optional) |

If `inspect_csvs.py` reports the wrong column, add `--audio_column X --text_column Y` to the `torchrun` line in `lanta/train_common.sh`.
- Relative paths are resolved against the CSV's own folder, or against `--audio_root` if you set it.
- If a CSV has `start`/`end` columns (rows that are segments of longer recordings), the script would train on whole files. In that case it needs a change.

## Defaults and knobs

Override any of these at submit time, e.g. `sbatch --export=ALL,LR=1e-5,EPOCHS=2 lanta/train_filtered_unrefined.sbatch`.

| Var | Default | Note |
|---|---|---|
| `EPOCHS` / `MAX_STEPS` | 1 / -1 | `MAX_STEPS > 0` takes priority |
| `LR` | 2e-5 | cosine schedule, 2% warmup, full fine-tune in bf16 |
| `BS` × `ACCUM` × GPUs | 16 × 2 × 4 = 128 | use the smoke test to try `BS=32` on the 40 GB A100s |
| `WORKERS` | 12 per GPU | audio decoding runs in DataLoader workers |

What the script does with the data:
- Clips over 30 s or under 0.3 s are skipped, as are transcripts over 448 tokens. Unreadable files are skipped and logged, and the script picks a random replacement row.
- Texts are used as-is ("unrefined"); only whitespace is collapsed.
- Loss is cross-entropy on the transcript tokens plus `<|im_end|>`. The audio prompt is masked out.

Pass `--freeze_audio_tower` to train only the projector and the LLM.
