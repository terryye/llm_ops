#!/usr/bin/env python3
"""
Assignment 1: data collection and preprocessing for foundation model pre-training.

This is the single entry point named in the assignment's deliverables. It runs
the four pipeline stages in order, each of which also runs standalone:

    1. src/data_collection.py   Hugging Face -> data/raw/     (sharded JSONL.gz)
    2. src/data_cleaning.py     data/raw     -> data/clean/   (sharded JSONL.gz)
    3. src/tokenize_corpus.py   data/clean   -> data/tokens/  (packed uint16)
    4. src/data_loader.py       data/tokens  -> sample_dataset.pt

Each stage writes a ``manifest.json`` next to its output recording inputs,
parameters, environment and per-stage statistics, so any stage can be audited
or re-run on its own without repeating the ones before it.

Usage
-----
    python data_collection_preprocessing.py                 # full run, ~1.5 GB
    python data_collection_preprocessing.py --smoke         # ~2 min end-to-end
    python data_collection_preprocessing.py --skip collect  # reuse data/raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STAGES = ("collect", "clean", "tokenize", "loader")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="root for stage outputs")
    p.add_argument(
        "--total-mb",
        type=float,
        default=1500.0,
        help="raw text budget. ~1.5x the 1 GB requirement so the corpus still "
        "clears 1 GB after Stage 2 removes duplicates and short documents.",
    )
    p.add_argument("--tokenizer", default="gpt2", help="any Hugging Face AutoTokenizer id")
    p.add_argument("--block-size", type=int, default=1024, help="tokens per training block")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--sample-out", type=Path, default=Path("sample_dataset.pt"))
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=STAGES,
        help="stages to skip because their output already exists",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end run (20 MB, capped documents) to validate the pipeline",
    )
    return p.parse_args(argv)


def banner(n: int, name: str) -> float:
    print(f"\n{'#' * 76}\n### Stage {n}: {name}\n{'#' * 76}", flush=True)
    return time.monotonic()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import data_cleaning
    import data_collection
    import data_loader
    import tokenize_corpus

    raw, clean, tokens = (args.data_dir / d for d in ("raw", "clean", "tokens"))
    total_mb = 20.0 if args.smoke else args.total_mb
    doc_limit = ["--limit", "8000"] if args.smoke else []
    timings: list[tuple[str, float]] = []

    if "collect" not in args.skip:
        t = banner(1, f"collect {total_mb:,.0f} MB of raw text -> {raw}")
        rc = data_collection.main(
            ["--out-dir", str(raw), "--total-mb", str(total_mb),
             "--seed", str(args.seed), "--force"]
        )
        if rc != 0:
            return rc
        timings.append(("collect", time.monotonic() - t))

    if "clean" not in args.skip:
        t = banner(2, f"clean, normalize and deduplicate -> {clean}")
        rc = data_cleaning.main(
            ["--in-dir", str(raw), "--out-dir", str(clean), "--force", *doc_limit]
        )
        if rc != 0:
            return rc
        timings.append(("clean", time.monotonic() - t))

    if "tokenize" not in args.skip:
        t = banner(3, f"tokenize with {args.tokenizer} -> {tokens}")
        rc = tokenize_corpus.main(
            ["--in-dir", str(clean), "--out-dir", str(tokens),
             "--tokenizer", args.tokenizer, "--block-size", str(args.block_size), "--force"]
        )
        if rc != 0:
            return rc
        timings.append(("tokenize", time.monotonic() - t))

    if "loader" not in args.skip:
        t = banner(4, f"build data loaders and write {args.sample_out}")
        rc = data_loader.main(
            ["--tokens-dir", str(tokens), "--clean-dir", str(clean),
             "--sample-out", str(args.sample_out), "--seed", str(args.seed)]
        )
        if rc != 0:
            return rc
        timings.append(("loader", time.monotonic() - t))

    print(f"\n{'=' * 76}\nPipeline complete.")
    for name, secs in timings:
        print(f"  {name:<12}{secs:>8,.0f}s")
    print(f"  {'TOTAL':<12}{sum(s for _, s in timings):>8,.0f}s")
    print("\nArtifacts:")
    for path in (raw, clean, tokens):
        man = path / "manifest.json"
        print(f"  {str(path) + '/':<16}{'manifest.json' if man.is_file() else '(missing)'}")
    if args.sample_out.is_file():
        print(f"  {str(args.sample_out):<16}  {args.sample_out.stat().st_size / 1024:,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
