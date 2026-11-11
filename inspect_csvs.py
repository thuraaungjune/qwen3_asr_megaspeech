#!/usr/bin/env python
"""Check the MegaSpeech CSVs before training. Run it on the LANTA frontend; it's cheap.

    python inspect_csvs.py /project/lt200217-thwhis/data/annotated_audios/original_of_megaspeech_v3/*/*.csv

For each CSV it prints the columns, the columns finetune_qwen3_asr.py would pick, a few
rows, how many audio paths exist (checked on a sample), and the duration totals.
"""
import argparse
import os
import random

import pandas as pd
import soundfile as sf

from finetune_qwen3_asr import AUDIO_COLS, DURATION_COLS, TEXT_COLS, _pick

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--audio_root", default=None)
ap.add_argument("--check", type=int, default=200, help="audio files to open per CSV")
args = ap.parse_args()

for path in args.csvs:
    print("=" * 100, f"\n{path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"rows: {len(df):,}   columns: {list(df.columns)}")
    try:
        ac = _pick(df.columns, None, AUDIO_COLS, "audio")
        tc = _pick(df.columns, None, TEXT_COLS, "text")
    except ValueError as e:
        print("  !!", e)
        print(df.head(3).to_string())
        continue
    dc = _pick(df.columns, None, DURATION_COLS, "duration", required=False)
    print(f"auto-detected -> audio={ac!r} text={tc!r} duration={dc!r}")
    extra = [c for c in df.columns if c.lower() in ("start", "end", "offset", "begin")]
    if extra:
        print(f"  !! segment columns {extra} found: the trainer reads WHOLE files; tell me if rows are segments of longer audio")
    print(df[[c for c in (ac, tc, dc) if c]].head(3).to_string(max_colwidth=90))

    root = args.audio_root or os.path.dirname(os.path.abspath(path))
    paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in df[ac].astype(str)]
    sample = random.Random(0).sample(paths, min(args.check, len(paths)))
    exist = [p for p in sample if os.path.exists(p)]
    readable, srs, durs = 0, {}, []
    for p in exist[:50]:
        try:
            info = sf.info(p)
            readable += 1
            srs[info.samplerate] = srs.get(info.samplerate, 0) + 1
            durs.append(info.duration)
        except Exception as e:
            print(f"  unreadable by soundfile: {p} ({e})")
    print(f"paths existing: {len(exist)}/{len(sample)} sampled | soundfile-readable: {readable}/{min(50, len(exist))}"
          f" | sample rates {srs}")
    if len(exist) < len(sample):
        print("  e.g. missing:", next(p for p in sample if not os.path.exists(p)))
    text = df[tc].astype(str)
    print(f"empty text: {(text.str.strip() == '').sum() + df[tc].isna().sum():,} | mean chars {text.str.len().mean():.0f}")
    if dc:
        d = pd.to_numeric(df[dc], errors="coerce")
        print(f"duration: {d.sum() / 3600:,.1f} h | mean {d.mean():.1f}s | max {d.max():.1f}s | >30s: {(d > 30).sum():,} rows")
    elif durs:
        print(f"no duration column; sampled files average {sum(durs) / len(durs):.1f}s (max {max(durs):.1f}s)")
