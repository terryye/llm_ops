"""
Assignment 1, Stage 3: tokenization.

Encodes the cleaned corpus with a transformer-compatible tokenizer and packs the
result into flat ``uint16`` token streams that Stage 4 memory-maps.

Design notes
------------
* **Packing, not padding.** Documents are concatenated with an end-of-text
  separator and then cut into fixed ``--block-size`` blocks. This is how GPT-2
  and every pretraining pipeline since has done it: padding to the longest
  document in a batch would waste a large fraction of every batch on <pad>,
  and the assignment's "handle sequences longer than the maximum block size via
  chunking" falls out of packing for free -- a 10k-token document simply spans
  ten blocks instead of being truncated to one.
* **uint16 on disk.** GPT-2's vocabulary is 50,257 entries, which fits in 16
  bits. Storing int64 (numpy's default) would quadruple a multi-GB artifact for
  no benefit. The loader widens to int64 only for the tokens in the current
  batch.
* **The validation split is per-document and hash-based**, not a tail slice.
  The corpus is written source-by-source, so holding out the last 0.5% of
  tokens would produce a validation set made entirely of general-web text.
  Hashing the document id keeps every domain represented and makes the split
  reproducible without storing an index.
* ``model_max_length`` is raised before encoding. It is a *generation-time*
  guard; leaving it at 1024 makes the tokenizer log a warning for every
  document longer than a block, which here is most of them.

Usage
-----
    python src/tokenize_corpus.py --in-dir data/clean --out-dir data/tokens
    python src/tokenize_corpus.py --in-dir data/clean --out-dir data/tokens --limit 5000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# The Rust tokenizer parallelises internally; leaving this unset makes
# transformers print a fork-safety warning on every batch.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from data_collection import load_env_file, make_progress  # noqa: E402  (after the env var)

# Same .env as Stage 1: the tokenizer is fetched from the Hub, and
# unauthenticated Hub requests are rate limited.
load_env_file()

TOKEN_DTYPE = np.uint16
SPLITS = ("train", "val")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_for(doc_id: str, val_per_mille: int) -> str:
    """Assign a document to train/val by a stable hash of its id.

    Python's built-in ``hash`` is salted per process, so it would produce a
    different split on every run; sha256 keeps the split reproducible.
    """
    bucket = int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:4], "big") % 1000
    return "val" if bucket < val_per_mille else "train"


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@dataclass
class SplitWriter:
    """Accumulates token ids and flushes whole blocks to a flat binary file."""

    path: Path
    block_size: int
    _fh: Any = None
    buffer: list[int] = field(default_factory=list)
    tokens_written: int = 0
    blocks_written: int = 0
    documents: int = 0
    tokens_dropped: int = 0

    def open(self) -> None:
        self._fh = open(self.path, "wb")

    def add(self, ids: list[int]) -> None:
        self.documents += 1
        self.buffer.extend(ids)
        if len(self.buffer) >= self.block_size * 1024:
            self._flush(keep_remainder=True)

    def _flush(self, keep_remainder: bool) -> None:
        n_blocks = len(self.buffer) // self.block_size
        if n_blocks:
            cut = n_blocks * self.block_size
            np.asarray(self.buffer[:cut], dtype=TOKEN_DTYPE).tofile(self._fh)
            self.tokens_written += cut
            self.blocks_written += n_blocks
            self.buffer = self.buffer[cut:]
        if not keep_remainder:
            # A trailing partial block is discarded rather than padded: one
            # short block in millions is not worth teaching the model a <pad>
            # token it will never see again.
            self.tokens_dropped += len(self.buffer)
            self.buffer = []

    def close(self) -> None:
        if self._fh is None:
            return
        self._flush(keep_remainder=False)
        self._fh.close()
        self._fh = None


def iter_clean_records(in_dir: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    shards = sorted(in_dir.glob("clean-*.jsonl.gz"))
    if not shards:
        raise SystemExit(f"error: no clean-*.jsonl.gz in {in_dir}; run data_cleaning.py first")
    seen = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
                seen += 1
                if limit is not None and seen >= limit:
                    return


def batched(it: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for rec in it:
        batch.append(rec)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_corpus(
    in_dir: Path,
    out_dir: Path,
    tokenizer,
    block_size: int,
    batch_size: int,
    val_per_mille: int,
    limit: int | None,
    progress,
) -> dict[str, Any]:
    eos_id = tokenizer.eos_token_id
    writers = {s: SplitWriter(out_dir / f"{s}.bin", block_size) for s in SPLITS}
    for w in writers.values():
        w.open()

    per_source: dict[str, int] = {}
    doc_token_lengths: list[int] = []
    started = time.monotonic()

    try:
        for batch in batched(iter_clean_records(in_dir, limit), batch_size):
            # One call per batch, not per document: the Rust tokenizer
            # parallelises across a batch but not across separate calls.
            encoded = tokenizer([r["text"] for r in batch], add_special_tokens=False)["input_ids"]
            for rec, ids in zip(batch, encoded):
                ids = ids + [eos_id]  # document boundary the model can learn
                writers[split_for(rec["id"], val_per_mille)].add(ids)
                per_source[rec["source"]] = per_source.get(rec["source"], 0) + len(ids)
                doc_token_lengths.append(len(ids))
                progress(rec["n_bytes"])
    finally:
        for w in writers.values():
            w.close()

    lengths = np.asarray(doc_token_lengths, dtype=np.int64)
    return {
        "seconds": round(time.monotonic() - started, 1),
        "splits": {
            s: {
                "path": writers[s].path.name,
                "documents": writers[s].documents,
                "tokens": writers[s].tokens_written,
                "blocks": writers[s].blocks_written,
                "tokens_dropped_in_partial_block": writers[s].tokens_dropped,
                "bytes_on_disk": writers[s].path.stat().st_size,
            }
            for s in SPLITS
        },
        "tokens_per_source": per_source,
        "document_token_length": {
            "count": int(lengths.size),
            "mean": float(lengths.mean()) if lengths.size else 0.0,
            "min": int(lengths.min()) if lengths.size else 0,
            "p50": float(np.percentile(lengths, 50)) if lengths.size else 0.0,
            "p90": float(np.percentile(lengths, 90)) if lengths.size else 0.0,
            "p99": float(np.percentile(lengths, 99)) if lengths.size else 0.0,
            "max": int(lengths.max()) if lengths.size else 0,
            "over_block_size": int((lengths > block_size).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize the cleaned corpus into packed uint16 blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-dir", type=Path, default=Path("data/clean"), help="Stage 2 output")
    p.add_argument("--out-dir", type=Path, default=Path("data/tokens"), help="token output")
    p.add_argument("--tokenizer", default="gpt2", help="any Hugging Face AutoTokenizer id")
    p.add_argument("--block-size", type=int, default=1024, help="tokens per training block")
    p.add_argument("--batch-size", type=int, default=512, help="documents per tokenizer call")
    p.add_argument(
        "--val-per-mille",
        type=int,
        default=5,
        help="documents per 1000 held out for validation",
    )
    p.add_argument("--limit", type=int, default=None, help="stop after N documents (smoke tests)")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"error: {out_dir} is not empty (pass --force to overwrite)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("*.bin"), out_dir / "manifest.json"]:
        stale.unlink(missing_ok=True)

    from transformers import AutoTokenizer

    print(f"Loading tokenizer {args.tokenizer!r}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise SystemExit(f"error: tokenizer {args.tokenizer!r} has no eos_token; pick another")
    # See the module docstring: this is a generation guard, not an encoding one.
    tokenizer.model_max_length = int(1e9)

    if tokenizer.vocab_size > np.iinfo(TOKEN_DTYPE).max:
        raise SystemExit(
            f"error: vocab {tokenizer.vocab_size} exceeds {TOKEN_DTYPE.__name__} range; "
            "widen TOKEN_DTYPE in tokenize_corpus.py and data_loader.py together"
        )

    clean_manifest_path = args.in_dir / "manifest.json"
    clean_manifest = (
        json.loads(clean_manifest_path.read_text()) if clean_manifest_path.is_file() else {}
    )
    total_bytes = clean_manifest.get("totals", {}).get("text_bytes", 0)

    print(f"Tokenizing {args.in_dir} -> {out_dir} (block {args.block_size})", file=sys.stderr)
    progress, close_progress, _log = make_progress(total_bytes)
    started = datetime.now(timezone.utc)
    try:
        result = tokenize_corpus(
            args.in_dir,
            out_dir,
            tokenizer,
            args.block_size,
            args.batch_size,
            args.val_per_mille,
            args.limit,
            progress,
        )
    finally:
        close_progress()

    manifest = {
        "stage": "03-tokenization",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "tokenizer": {
            "id": args.tokenizer,
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "is_fast": tokenizer.is_fast,
        },
        "format": {
            "dtype": TOKEN_DTYPE.__name__,
            "block_size": args.block_size,
            "layout": "flat concatenated token ids; block i is tokens [i*B, (i+1)*B)",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "input": {"text_bytes": total_bytes, "documents": clean_manifest.get("totals", {}).get("documents")},
        **result,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print_summary(manifest)
    return 0 if manifest["splits"]["train"]["blocks"] > 0 else 1


def print_summary(manifest: dict[str, Any]) -> None:
    tok, fmt = manifest["tokenizer"], manifest["format"]
    print("\n" + "=" * 68)
    print(f"tokenizer {tok['id']} ({tok['class']}, vocab {tok['vocab_size']:,}, fast={tok['is_fast']})")
    print(f"format    {fmt['dtype']}, block_size {fmt['block_size']}")
    print("-" * 68)
    print(f"{'split':<10}{'docs':>12}{'tokens':>16}{'blocks':>12}{'MB on disk':>14}")
    for name, s in manifest["splits"].items():
        print(
            f"{name:<10}{s['documents']:>12,}{s['tokens']:>16,}{s['blocks']:>12,}"
            f"{s['bytes_on_disk'] / 1e6:>14,.1f}"
        )
    print("-" * 68)
    print("tokens per source:")
    for src, n in sorted(manifest["tokens_per_source"].items(), key=lambda kv: -kv[1]):
        print(f"  {src:<14}{n:>16,}")
    d = manifest["document_token_length"]
    print("-" * 68)
    print(
        f"doc length (tokens): mean {d['mean']:,.0f}  p50 {d['p50']:,.0f}  "
        f"p90 {d['p90']:,.0f}  p99 {d['p99']:,.0f}  max {d['max']:,}"
    )
    print(
        f"{d['over_block_size']:,} of {d['count']:,} documents "
        f"({100 * d['over_block_size'] / max(d['count'], 1):.1f}%) exceed one block "
        "and are split across blocks by packing"
    )
    total_tokens = sum(s["tokens"] for s in manifest["splits"].values())
    print(f"total {total_tokens:,} tokens in {manifest['seconds']:,.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
