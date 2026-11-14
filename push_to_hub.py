#!/usr/bin/env python
"""Upload a trained run to the Hugging Face Hub as a PUBLIC repo with MANUAL access approval (gated="manual").

Run it on the LANTA frontend (compute nodes have no internet):

    export HF_TOKEN=hf_xxx        # a WRITE token; or run `hf auth login` once instead
    python push_to_hub.py --run_dir checkpoints/qwen3-asr-0.6b-megaspeech-unfiltered-unrefined \
                          --repo_id ThuraAung1601/qwen3-asr-0.6b-megaspeech-unfiltered-unrefined

By default it uploads <run_dir>/final (the merged model for LoRA runs). If training hasn't finished,
pass --checkpoint latest (or checkpoint-12000) to upload an intermediate full-model checkpoint.
Optimizer / scheduler / RNG state are never uploaded.
"""
import argparse
import glob
import os
import re
import sys

from huggingface_hub import CommitOperationAdd, HfApi

PROCESSOR_FILES = ["tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja",
                   "preprocessor_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "generation_config.json"]
SKIP = re.compile(r"(optimizer|scheduler|rng_state.*|trainer_state|training_args|scaler)\.(pt|pth|bin|json)$")

DATA_DESC = {
    "unfiltered-unrefined": "MegaSpeech v3 **unfiltered audios + unrefined texts** (full original): gigaspeech-train, scraped, "
                            "yodas (unfiltered), plus ai-cooking and thai-central (manually created, never filtered).",
    "filtered-unrefined": "MegaSpeech v3 **filtered audios + unrefined texts**: gigaspeech-train, scraped, yodas (filtered), "
                          "plus ai-cooking and thai-central (manually created, never filtered).",
}


def pick_model_dir(run_dir: str, checkpoint: str) -> str:
    if checkpoint == "latest":
        cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        if not cks:
            sys.exit(f"No checkpoint-* folders in {run_dir}")
        path = cks[-1]
    else:
        path = os.path.join(run_dir, checkpoint)
    if not os.path.isdir(path):
        sys.exit(f"{path} does not exist. Training not finished? Try --checkpoint latest "
                 f"(available: {sorted(os.listdir(run_dir))})")
    if os.path.exists(os.path.join(path, "adapter_config.json")) and not os.path.exists(os.path.join(path, "model.safetensors")):
        sys.exit(f"{path} is a LoRA adapter-only checkpoint. Upload the merged {run_dir}/final instead, "
                 "or merge it first (PeftModel.from_pretrained(base, path).merge_and_unload()).")
    if not glob.glob(os.path.join(path, "model*.safetensors")):
        sys.exit(f"No model*.safetensors in {path}")
    return path


def final_cer(run_dir: str):
    pred = os.path.join(run_dir, "final_eval_predictions.csv")
    if not os.path.exists(pred):
        return None, 0
    try:
        import jiwer
        import pandas as pd
        d = pd.read_csv(pred).fillna("")
        strip = lambda s: "".join(str(s).split())
        d = d[d.ref.map(strip) != ""]
        return 100 * jiwer.cer(d.ref.map(strip).tolist(), d.hyp.map(strip).tolist()), len(d)
    except Exception as e:
        print("could not compute CER from predictions:", e)
        return None, 0


def model_card(repo_id: str, run_name: str, model_dir: str, cer, n_cer: int, base_model: str) -> str:
    variant = next((k for k in DATA_DESC if k in run_name), None)
    lora = "lora" in run_name
    cer_line = (f"| gigaspeech-dev (random {n_cer} utterances, greedy) | CER (spaces removed) | {cer:.2f}% |"
                if cer is not None else "| gigaspeech-dev | CER | see evaluation |")
    return f"""---
language:
- th
pipeline_tag: automatic-speech-recognition
library_name: transformers
base_model: {base_model}
tags:
- qwen3-asr
- thai
- speech-recognition
extra_gated_prompt: "Access is reviewed manually. Please describe your intended use."
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# {repo_id.split('/')[-1]}

[{base_model}](https://huggingface.co/{base_model}) fine-tuned for Thai ASR
({'LoRA r=64 on the LLM + trained projector, merged into the base weights' if lora else 'full fine-tuning'}).

**Training data:** {DATA_DESC.get(variant, 'MegaSpeech v3')}
Clips longer than 30 s were skipped. Transcripts were used as-is (only whitespace collapsed).

**Uploaded from:** `{os.path.basename(model_dir)}`

## Results

| Eval set | Metric | Value |
|---|---|---|
{cer_line}

## Usage

```python
import librosa, torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

repo = "{repo_id}"
processor = AutoProcessor.from_pretrained(repo)
model = Qwen3ASRForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

audio, _ = librosa.load("audio.wav", sr=16000)
inputs = processor.apply_transcription_request(audio=audio, language="th", return_tensors="pt").to(model.device, model.dtype)
out = model.generate(**inputs, max_new_tokens=448, do_sample=False)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For audio longer than ~30 s, split it into chunks first; the model was trained on clips of 30 s or less.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="checkpoints/qwen3-asr-0.6b-megaspeech-unfiltered-unrefined")
    ap.add_argument("--repo_id", default=None, help="e.g. ThuraAung1601/qwen3-asr-0.6b-megaspeech-unfiltered-unrefined "
                                                     "(default: <your HF user>/<run folder name>)")
    ap.add_argument("--checkpoint", default="final", help="'final', 'latest', or a folder name like checkpoint-12000")
    ap.add_argument("--base_model", default="Qwen/Qwen3-ASR-0.6B-hf")
    ap.add_argument("--gated", default="manual", choices=["manual", "auto", "none"])
    ap.add_argument("--token", default=None, help="default: $HF_TOKEN, else the token saved by `hf auth login`")
    ap.add_argument("--no_card", action="store_true", help="don't write a README.md model card")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as e:
        sys.exit(f"Not logged in to the Hub ({e}). export HF_TOKEN=<write token> or run `hf auth login`.")

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    run_name = os.path.basename(run_dir)
    repo_id = args.repo_id or f"{user}/{run_name}"
    model_dir = pick_model_dir(run_dir, args.checkpoint)

    # model files from model_dir; processor files from model_dir, falling back to run_dir (saved there at start)
    ops, names = [], set()
    for f in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p) and not SKIP.search(f) and f != "README.md":
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=p)); names.add(f)
    for f in PROCESSOR_FILES:
        p = os.path.join(run_dir, f)
        if f not in names and os.path.isfile(p):
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=p)); names.add(f)
    missing = [f for f in ("config.json", "tokenizer.json", "processor_config.json") if f not in names]
    if missing:
        sys.exit(f"Missing {missing} - the repo would not load. Check {model_dir} and {run_dir}.")
    if not args.no_card:
        cer, n = final_cer(run_dir) if args.checkpoint == "final" else (None, 0)
        card = model_card(repo_id, run_name, model_dir, cer, n, args.base_model)
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
        names.add("README.md")

    size = sum(os.path.getsize(op.path_or_fileobj) for op in ops if isinstance(op.path_or_fileobj, str)) / 1e9
    print(f"Logged in as {user}\nUploading {model_dir}\n  -> https://huggingface.co/{repo_id}  ({size:.2f} GB)")
    for n in sorted(names):
        print("   ", n)

    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    gated = False if args.gated == "none" else args.gated
    api.update_repo_settings(repo_id, gated=gated, private=False)   # public, access requests approved by hand
    info = api.create_commit(repo_id=repo_id, operations=ops,
                             commit_message=f"Upload {run_name} ({os.path.basename(model_dir)})")
    print(f"\nDone: {info.commit_url}\nVisibility: public | gated: {args.gated} "
          f"(review requests at https://huggingface.co/{repo_id}/settings)")


if __name__ == "__main__":
    main()
