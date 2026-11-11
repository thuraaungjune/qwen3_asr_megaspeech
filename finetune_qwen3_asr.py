#!/usr/bin/env python
"""Supervised fine-tuning of Qwen3-ASR (default: Qwen/Qwen3-ASR-0.6B-hf) on CSV manifests.

Uses the Hugging Face `Trainer` with plain cross-entropy on the transcript tokens.

Every CSV row is one utterance: a path to an audio file plus its transcript. Columns
are auto-detected (see AUDIO_COLS / TEXT_COLS / DURATION_COLS) or set explicitly
with --audio_column / --text_column / --duration_column. Relative audio paths are
resolved against --audio_root, or against the CSV's own folder if that is not set.

The training sequence is the processor's own transcription prompt followed by the
target, and only the target tokens are scored:

    <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    <|im_start|>assistant\nlanguage Thai<asr_text>  {transcript}<|im_end|>

Single GPU:   python finetune_qwen3_asr.py --train_csvs a.csv b.csv --output_dir out --bf16
Multi GPU:    torchrun --nproc_per_node 4 finetune_qwen3_asr.py ...   (any TrainingArguments flag works)
Re-running with the same --output_dir resumes from the newest checkpoint-* in it.
"""
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen3ASRForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

SR = 16000
AUDIO_COLS = ["audio_path", "audio_filepath", "path", "filepath", "file_path", "audio", "wav", "wav_path", "file", "filename"]
TEXT_COLS = ["text", "transcription", "transcript", "sentence", "label", "normalized_text", "raw_text"]
DURATION_COLS = ["duration", "duration_s", "duration_sec", "seconds", "length", "audio_duration"]


# ----------------------------------------------------------------------------- args
@dataclass
class DataArgs:
    train_csvs: List[str] = field(metadata={"help": "One or more training CSVs."})
    eval_csvs: List[str] = field(default_factory=list, metadata={"help": "CSVs used for eval loss / final CER."})
    model_name_or_path: str = "Qwen/Qwen3-ASR-0.6B-hf"
    audio_column: Optional[str] = None
    text_column: Optional[str] = None
    duration_column: Optional[str] = None
    audio_root: Optional[str] = field(default=None, metadata={"help": "Base dir for relative audio paths (default: the CSV's folder)."})
    language: str = field(default="th", metadata={"help": "Language tag in the prompt; 'th' -> 'language Thai<asr_text>'."})
    min_audio_seconds: float = 0.3
    max_audio_seconds: float = 30.0
    max_label_tokens: int = field(default=448, metadata={"help": "Rows whose transcript is longer than this are skipped (never truncated)."})
    eval_max_samples: int = field(default=2000, metadata={"help": "Random subset of the eval CSVs used during training (0 = all)."})
    final_cer_samples: int = field(default=500, metadata={"help": "Eval rows decoded with greedy generate after training (0 = skip)."})
    final_cer_batch_size: int = 16
    freeze_audio_tower: bool = field(default=False, metadata={"help": "Full fine-tune: train only the projector + LLM."})
    use_lora: bool = field(default=False, metadata={"help": "LoRA on the LLM (base frozen); merged into final/ at the end."})
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_audio_tower: bool = field(default=False, metadata={"help": "Also put LoRA on the audio encoder."})
    lora_train_projector: bool = field(default=True, metadata={"help": "Fully train the small audio->LLM projector alongside LoRA."})
    max_train_samples: int = field(default=0, metadata={"help": "Debug: cap on training rows (0 = all)."})


# ----------------------------------------------------------------------------- data
def _pick(cols, wanted: Optional[str], candidates: List[str], what: str, required=True) -> Optional[str]:
    if wanted:
        if wanted not in cols:
            raise ValueError(f"--{what}_column={wanted!r} not in CSV columns {list(cols)}")
        return wanted
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    if required:
        raise ValueError(f"Could not auto-detect the {what} column among {list(cols)}; pass --{what}_column")
    return None


def normalize_text(t) -> str:
    return " ".join(str(t).split()) if isinstance(t, str) else ""


def load_manifest(csvs: List[str], a: DataArgs, rank0: bool) -> pd.DataFrame:
    frames = []
    for path in csvs:
        df = pd.read_csv(path, low_memory=False)
        ac = _pick(df.columns, a.audio_column, AUDIO_COLS, "audio")
        tc = _pick(df.columns, a.text_column, TEXT_COLS, "text")
        dc = _pick(df.columns, a.duration_column, DURATION_COLS, "duration", required=False)
        root = a.audio_root or os.path.dirname(os.path.abspath(path))
        out = pd.DataFrame({
            "audio": [p if os.path.isabs(p) else os.path.join(root, p) for p in df[ac].astype(str)],
            "text": df[tc].map(normalize_text),
            "duration": pd.to_numeric(df[dc], errors="coerce") if dc else np.nan,
            "source": os.path.splitext(os.path.basename(path))[0],
        })
        n0 = len(out)
        out = out[out.text.str.len() > 0]
        if dc:
            ok = out.duration.isna() | out.duration.between(a.min_audio_seconds, a.max_audio_seconds)
            out = out[ok]
        hours = out.duration.sum() / 3600 if dc else float("nan")
        if rank0:
            print(f"  {path}\n    columns: audio={ac!r} text={tc!r} duration={dc!r} | rows {n0:,} -> {len(out):,}"
                  f" | {hours:,.1f} h" + ("" if dc else " (no duration column: lengths checked while loading)"))
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


class AsrCsvDataset(Dataset):
    """Loads audio lazily. A row that can't be read or falls outside the length limits is replaced by a random other row."""

    def __init__(self, df: pd.DataFrame, tokenizer, a: DataArgs, end_id: int):
        self.audio, self.text = df.audio.tolist(), df.text.tolist()
        self.tok, self.a, self.end_id = tokenizer, a, end_id
        self.n_bad = 0

    def __len__(self):
        return len(self.audio)

    def _load(self, path: str) -> np.ndarray:
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        except Exception:
            wav, sr = librosa.load(path, sr=None, mono=True)   # formats libsndfile can't read
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)

    def __getitem__(self, i):
        for attempt in range(20):
            try:
                wav = self._load(self.audio[i])
                dur = len(wav) / SR
                if not (self.a.min_audio_seconds <= dur <= self.a.max_audio_seconds):
                    raise ValueError(f"duration {dur:.1f}s")
                ids = self.tok(self.text[i], add_special_tokens=False)["input_ids"]
                if not ids or len(ids) > self.a.max_label_tokens:
                    raise ValueError(f"{len(ids)} label tokens")
                return {"wav": wav, "labels": ids + [self.end_id], "text": self.text[i]}
            except Exception as e:
                self.n_bad += 1
                if self.n_bad <= 20 or self.n_bad % 1000 == 0:
                    print(f"[data] skipping {self.audio[i]} ({type(e).__name__}: {e}) - {self.n_bad} skipped in this worker",
                          file=sys.stderr)
                i = random.randrange(len(self.audio))
        raise RuntimeError("20 unreadable rows in a row - check the audio paths with inspect_csvs.py")


@dataclass
class Collator:
    processor: object
    language: str

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        enc = self.processor.apply_transcription_request(
            audio=[b["wav"] for b in batch], language=[self.language] * len(batch), return_tensors="pt")
        pad_id = self.processor.tokenizer.pad_token_id
        rows, labs = [], []
        for i, b in enumerate(batch):
            prompt = enc["input_ids"][i][enc["attention_mask"][i].bool()]      # works for left or right padding
            comp = torch.tensor(b["labels"], dtype=torch.long)
            rows.append(torch.cat([prompt, comp]))
            labs.append(torch.cat([torch.full_like(prompt, -100), comp]))
        L = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
        labels = torch.full((len(rows), L), -100, dtype=torch.long)
        attn = torch.zeros((len(rows), L), dtype=torch.long)
        for i, (r, l) in enumerate(zip(rows, labs)):
            input_ids[i, : len(r)], labels[i, : len(r)], attn[i, : len(r)] = r, l, 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
                "input_features": enc["input_features"], "input_features_mask": enc["input_features_mask"]}


# ----------------------------------------------------------------------------- trainer
class AsrTrainer(Trainer):
    """Cross-entropy on the target tokens only. Logits are computed just for the positions from the
    first target onward (the audio prompt is ~13 tokens/s, so this saves most of the 151k-vocab logit memory)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        first = int((labels != -100).float().argmax(dim=1).min())     # earliest target position in the batch
        n_keep = labels.shape[1] - (first - 1)                         # logits for positions first-1 .. L-1
        out = model(**inputs, logits_to_keep=n_keep, use_cache=False)
        logits = out.logits[:, :-1].float()                             # predict tokens first .. L-1
        target = labels[:, first:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)
        return (loss, out) if return_outputs else loss


def char_error_rate(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    strip = lambda s: "".join(s.split())       # Thai has no word spaces -> score characters without spaces
    pairs = [(strip(r), strip(h)) for r, h in zip(refs, hyps) if strip(r)]
    return 100 * jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])


@torch.no_grad()
def final_cer(model, processor, ds: AsrCsvDataset, a: DataArgs, out_dir: str):
    model.eval()
    tok = processor.tokenizer
    old_side, tok.padding_side = tok.padding_side, "left"                # batched generate needs left padding
    idx = random.Random(0).sample(range(len(ds)), min(a.final_cer_samples, len(ds)))
    refs, hyps = [], []
    for s in range(0, len(idx), a.final_cer_batch_size):
        items = [ds[i] for i in idx[s: s + a.final_cer_batch_size]]
        enc = processor.apply_transcription_request(
            audio=[b["wav"] for b in items], language=[a.language] * len(items), return_tensors="pt"
        ).to(model.device)
        enc["input_features"] = enc["input_features"].to(model.dtype)
        gen = model.generate(**enc, max_new_tokens=a.max_label_tokens + 8, do_sample=False, num_beams=1)
        hyps += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        refs += [b["text"] for b in items]
    tok.padding_side = old_side
    hyps = [h.strip() for h in hyps]
    cer = char_error_rate(refs, hyps)
    pd.DataFrame({"ref": refs, "hyp": hyps}).to_csv(os.path.join(out_dir, "final_eval_predictions.csv"), index=False)
    print(f"\nFinal greedy CER on {len(refs)} eval rows: {cer:.2f}%  (predictions -> final_eval_predictions.csv)")
    return cer


# ----------------------------------------------------------------------------- main
def main():
    parser = HfArgumentParser((DataArgs, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        a, targs = parser.parse_yaml_file(sys.argv[1]) if sys.argv[1].endswith(("yaml", "yml")) else parser.parse_json_file(sys.argv[1])
    else:
        a, targs = parser.parse_args_into_dataclasses()
    targs.remove_unused_columns = False            # the collator needs the raw 'wav' / 'labels' fields
    targs.prediction_loss_only = True
    targs.label_names = ["labels"]
    set_seed(targs.seed)
    rank0 = targs.process_index == 0

    processor = AutoProcessor.from_pretrained(a.model_name_or_path)
    tok = processor.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    if rank0:
        print("Training data:")
    train_df = load_manifest(a.train_csvs, a, rank0)
    if a.max_train_samples:
        train_df = train_df.sample(n=min(a.max_train_samples, len(train_df)), random_state=targs.seed)
    if rank0:
        print(f"  TOTAL train rows: {len(train_df):,}")
        print(train_df.groupby("source").agg(rows=("text", "size"), hours=("duration", lambda d: d.sum() / 3600)).to_string())
    train_ds = AsrCsvDataset(train_df, tok, a, end_id)

    eval_ds = None
    if a.eval_csvs:
        if rank0:
            print("Eval data:")
        eval_df = load_manifest(a.eval_csvs, a, rank0)
        if a.eval_max_samples and len(eval_df) > a.eval_max_samples:
            eval_df = eval_df.sample(n=a.eval_max_samples, random_state=0)
        eval_ds = AsrCsvDataset(eval_df.reset_index(drop=True), tok, a, end_id)
        if rank0:
            print(f"  eval rows: {len(eval_ds):,}")
    elif targs.eval_strategy != "no":
        targs.eval_strategy = "no"

    dtype = torch.bfloat16 if targs.bf16 else (torch.float16 if targs.fp16 else torch.float32)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        a.model_name_or_path, dtype=torch.float32 if targs.fp16 else dtype)   # fp16 AMP needs fp32 master weights
    model.config.use_cache = False
    if a.use_lora:
        from peft import LoraConfig, get_peft_model
        targets = [r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"]
        if a.lora_audio_tower:
            targets.append(r".*audio_tower\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|fc1|fc2)")
        model = get_peft_model(model, LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
            target_modules="|".join(f"(?:{t})" for t in targets),
            modules_to_save=["multi_modal_projector"] if a.lora_train_projector else None,
        ))
        if targs.gradient_checkpointing:
            model.enable_input_require_grads()     # frozen embeddings -> make checkpointed blocks see a grad-requiring input
    elif a.freeze_audio_tower:
        for n, p in model.named_parameters():
            if "audio_tower" in n:
                p.requires_grad_(False)
    if rank0:
        n_all = sum(p.numel() for p in model.parameters())
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model {a.model_name_or_path}: {n_all / 1e6:.0f}M params, {n_tr / 1e6:.0f}M trainable, dtype {dtype}")

    trainer = AsrTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                         data_collator=Collator(processor, a.language))
    trainer.model_accepts_loss_kwargs = False      # compute_loss returns a per-batch mean -> Trainer divides by grad-accum

    last = get_last_checkpoint(targs.output_dir) if os.path.isdir(targs.output_dir) else None
    if rank0:
        processor.save_pretrained(targs.output_dir)
        print(f"Resuming from {last}" if last else "Starting from scratch")
    trainer.train(resume_from_checkpoint=last)

    final_dir = os.path.join(targs.output_dir, "final")
    final_model = trainer.model
    if a.use_lora:
        trainer.save_model(os.path.join(targs.output_dir, "final_adapter"))   # adapter only (small), kept for reference
        if trainer.is_world_process_zero():
            final_model = trainer.model.merge_and_unload()                   # plain Qwen3ASRForConditionalGeneration
            final_model.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(final_dir)
        print(f"Saved final {'merged LoRA ' if a.use_lora else ''}model + processor -> {final_dir}")
        if eval_ds is not None and a.final_cer_samples:
            cer = final_cer(final_model, processor, eval_ds, a, targs.output_dir)
            trainer.log({"final_eval_cer": cer})


if __name__ == "__main__":
    main()
