"""
Assignment 1, Stage 1: raw corpus collection.

Streams a byte-budgeted, provenance-tagged sample from several public Hugging Face
datasets and writes it to sharded, gzipped JSONL plus a reproducibility manifest.

Design notes
------------
* Streaming (``streaming=True``) is used everywhere so we never download a 20 GB
  corpus just to keep 500 MB of it.
* Sampling is *stratified by shard*: we walk the dataset's parquet shards in a
  seeded random order and take an equal byte quota from each. Wikipedia's shards
  are alphabetical, so a plain sequential read would return only "A..." articles.
  ``IterableDataset.shuffle()`` would also fix that, but it fans out across ~10
  shards concurrently and buffers Arrow tables rather than rows -- on this corpus
  that means many GB of RSS and a storm of parallel Hub connections that the
  server drops. One shard at a time is one connection at a time, and RSS stays
  flat.
* The budget is measured in UTF-8 *bytes of text*, not in rows, because the
  assignment's size requirement ("at least 1GB of raw text") is a size requirement.
* No cleaning, filtering or deduplication happens here beyond dropping empty
  documents. Stage 2 owns that, and it needs the raw duplicates and noise intact
  in order to measure its own effect.
* Every record keeps its ``source``/``domain`` tag so later stages can report
  domain coverage and cross-source overlap.

Usage
-----
    python src/data_collection.py --out-dir data/raw --total-mb 1500
    python src/data_collection.py --out-dir data/smoke --total-mb 20 --sources wikipedia
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_env_file(explicit: Path | None = None) -> tuple[Path | None, list[str]]:
    """Load ``KEY=VALUE`` pairs from a .env file into os.environ.

    Looks next to this script, then in the working directory. Values already
    present in the environment win, so `HF_TOKEN=... python ...` still overrides.
    Returns the file used and the names (never the values) of the keys applied.

    An HF token is not required for these public datasets, but unauthenticated
    Hub requests are rate limited and noticeably slower.
    """
    candidates = [explicit] if explicit else [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        applied = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
                applied.append(key)
        return path, applied
    return None, []


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# `revision` pins the dataset repo commit so a re-run fetches identical data.
# Verified against the Hugging Face Hub API on 2026-09-18.


@dataclass(frozen=True)
class Source:
    name: str  # short tag stored on every record
    domain: str  # the assignment's diversity axis
    hf_id: str
    config: str
    split: str
    revision: str
    license: str
    weight: float  # share of the total byte budget
    text_field: str = "text"
    url_field: str | None = "url"
    title_field: str | None = None


SOURCES: tuple[Source, ...] = (
    Source(
        name="wikipedia",
        domain="encyclopedic",
        hf_id="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        license="CC-BY-SA-3.0 / GFDL",
        weight=1 / 3,
        title_field="title",
    ),
    Source(
        name="cc_news",
        domain="news",
        hf_id="vblagoje/cc_news",
        config="plain_text",
        split="train",
        revision="81eb2ce0d2a9dad6ad16b68ef750ec290880fa36",
        license="unknown (CC-News derivative); research use only",
        weight=1 / 3,
        title_field="title",
    ),
    Source(
        name="fineweb",
        domain="general_web",
        hf_id="HuggingFaceFW/fineweb",
        config="sample-10BT",
        split="train",
        revision="9bb295ddab0e05d785b879661af7260fed5140fc",
        license="ODC-BY-1.0",
        weight=1 / 3,
    ),
    # Opt-in alternative for the general-web slot. C4's `en.noclean` config is
    # deliberately unfiltered, which makes Stage 2's cleaning effect measurable
    # instead of marginal. Select it with `--sources wikipedia cc_news c4_noclean`.
    Source(
        name="c4_noclean",
        domain="general_web",
        hf_id="allenai/c4",
        config="en.noclean",
        split="train",
        revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        license="ODC-BY-1.0",
        weight=1 / 3,
    ),
)

SOURCES_BY_NAME = {s.name: s for s in SOURCES}
DEFAULT_SOURCES = ("wikipedia", "cc_news", "fineweb")


# ---------------------------------------------------------------------------
# Sharded JSONL writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Writes JSONL records into gzipped shards of a bounded size.

    Sharding keeps Stage 2 parallelisable (one worker per shard) and means a
    crash costs one shard rather than the whole crawl.
    """

    def __init__(self, out_dir: Path, prefix: str, shard_bytes: int, compresslevel: int = 6):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.compresslevel = compresslevel
        self.shard_index = -1
        self._fh: io.TextIOWrapper | None = None
        self._path: Path | None = None
        self._uncompressed = 0
        self._records = 0
        self.shards: list[dict[str, Any]] = []

    def _rotate(self) -> None:
        self.close()
        self.shard_index += 1
        self._path = self.out_dir / f"{self.prefix}-{self.shard_index:05d}.jsonl.gz"
        self._fh = gzip.open(self._path, "wt", encoding="utf-8", compresslevel=self.compresslevel)
        self._uncompressed = 0
        self._records = 0

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None or self._uncompressed >= self.shard_bytes:
            self._rotate()
        line = json.dumps(record, ensure_ascii=False) + "\n"
        assert self._fh is not None
        self._fh.write(line)
        self._uncompressed += len(line.encode("utf-8"))
        self._records += 1

    def close(self) -> None:
        if self._fh is None or self._path is None:
            return
        self._fh.close()
        self.shards.append(
            {
                "path": self._path.name,
                "records": self._records,
                "uncompressed_bytes": self._uncompressed,
                "compressed_bytes": self._path.stat().st_size,
                "sha256": _sha256_file(self._path),
            }
        )
        self._fh = None
        self._path = None


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


# Smallest byte quota worth paying a shard's connection setup for.
MIN_SHARD_QUOTA_BYTES = 8_000_000


@dataclass
class SourceStats:
    source: str
    domain: str
    hf_id: str
    config: str
    revision: str
    license: str
    target_bytes: int
    text_bytes: int = 0
    documents: int = 0
    empty_skipped: int = 0
    shards_total: int = 0
    shards_used: int = 0
    attempts: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _load_stream(src: Source):
    """Open the streaming dataset for one source. Cheap: no rows are read yet."""
    from datasets import load_dataset

    return load_dataset(
        src.hf_id,
        src.config,
        split=src.split,
        revision=src.revision,
        streaming=True,
    )


def _shard_stream(ds, n_shards: int, index: int, skip: int):
    """Iterate a single shard, skipping `skip` rows already consumed from it.

    Shards cannot seek, so a retry replays the shard's prefix. That costs at most
    one shard's quota, not the whole source, which is why retries are scoped here
    rather than around the source as a whole.
    """
    return itertools.islice(iter(ds.shard(num_shards=n_shards, index=index)), skip, None)


def collect_source(
    src: Source,
    writer: ShardWriter,
    target_bytes: int,
    seed: int,
    max_retries: int,
    progress,
    log,
) -> SourceStats:
    stats = SourceStats(
        source=src.name,
        domain=src.domain,
        hf_id=src.hf_id,
        config=src.config,
        revision=src.revision,
        license=src.license,
        target_bytes=target_bytes,
    )
    started = time.monotonic()

    try:
        ds = _load_stream(src)
    except Exception as exc:  # gated repo, bad revision, Hub outage
        stats.errors.append(f"load_dataset: {type(exc).__name__}: {exc}")
        log(f"  ! {src.name}: {stats.errors[-1]}")
        stats.seconds = round(time.monotonic() - started, 1)
        return stats

    stats.shards_total = ds.num_shards
    order = list(range(stats.shards_total))
    random.Random(seed).shuffle(order)
    # Opening a shard costs a fresh HTTPS connection plus a parquet footer read
    # (~2s), so a small budget spread over every shard is all setup and no data.
    # Full runs are unaffected: 500 MB / 8 MB is more shards than any source has.
    order = order[: max(1, min(len(order), target_bytes // MIN_SHARD_QUOTA_BYTES))]

    for pos, shard_idx in enumerate(order):
        remaining = target_bytes - stats.text_bytes
        if remaining <= 0:
            break
        # Self-balancing quota: recomputed each shard, so a shard that runs short
        # (or fails outright) is automatically made up by the ones after it.
        quota = math.ceil(remaining / (len(order) - pos))
        stats.shards_used += 1
        taken = 0
        rows = 0  # includes empties, so a retry resumes at the right offset

        for attempt in range(1, max_retries + 2):
            stats.attempts += 1
            try:
                for row in _shard_stream(ds, stats.shards_total, shard_idx, skip=rows):
                    rows += 1
                    text = row.get(src.text_field) or ""
                    if not text.strip():
                        stats.empty_skipped += 1
                        continue

                    raw = text.encode("utf-8")
                    record = {
                        "id": f"{src.name}:{stats.documents:08d}",
                        "source": src.name,
                        "domain": src.domain,
                        "shard": shard_idx,
                        "url": row.get(src.url_field) if src.url_field else None,
                        "title": row.get(src.title_field) if src.title_field else None,
                        # Hashing the untouched text here makes Stage 2's exact-dedup
                        # pass a cheap set lookup instead of a second full read.
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "n_bytes": len(raw),
                        "text": text,
                    }
                    writer.write(record)
                    stats.documents += 1
                    stats.text_bytes += len(raw)
                    taken += len(raw)
                    progress(len(raw))

                    if taken >= quota:
                        break
                break  # quota met, or the shard ran out; either way move on
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # network flake, parquet hiccup, ...
                msg = f"shard {shard_idx} attempt {attempt}: {type(exc).__name__}: {exc}"
                stats.errors.append(msg)
                log(f"  ! {src.name}: {msg}")
                if attempt > max_retries:
                    break  # abandon this shard; its quota rolls into the next one
                time.sleep(min(30, 2**attempt))

    if stats.text_bytes < target_bytes:
        stats.errors.append(
            f"exhausted {stats.shards_used} shard(s) at {stats.text_bytes} "
            f"of {target_bytes} bytes"
        )

    stats.seconds = round(time.monotonic() - started, 1)
    return stats


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def make_progress(total_bytes: int):
    """Return (update, close, log). Uses tqdm when available, else a plain ticker.

    ``log`` routes messages through tqdm so status lines do not shred the bar.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        state = {"done": 0, "last": 0.0}

        def cb(n: int) -> None:
            state["done"] += n
            now = time.monotonic()
            if now - state["last"] >= 5.0:
                state["last"] = now
                pct = 100 * state["done"] / total_bytes if total_bytes else 0
                print(f"  {state['done'] / 1e6:,.0f} MB / {total_bytes / 1e6:,.0f} MB ({pct:.1f}%)")

        return cb, (lambda: None), (lambda m: print(m, file=sys.stderr, flush=True))

    bar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1000, desc="collected")
    return bar.update, bar.close, (lambda m: tqdm.write(m, file=sys.stderr))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect a byte-budgeted, multi-domain raw text corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/raw"), help="output directory")
    p.add_argument(
        "--total-mb",
        type=float,
        default=1500.0,
        help="total raw text budget in MB (decimal). Aim ~1.5x the 1 GB "
        "requirement so the corpus still clears 1 GB after Stage 2 filtering.",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=[s.name for s in SOURCES],
        help="which sources to draw from; the budget is split by their weights",
    )
    p.add_argument("--shard-mb", type=float, default=128.0, help="uncompressed JSONL per shard")
    p.add_argument(
        "--seed",
        type=int,
        default=20260918,
        help="seed for the shard visiting order (reproducibility)",
    )
    p.add_argument("--max-retries", type=int, default=3, help="retries per shard on stream error")
    p.add_argument("--gzip-level", type=int, default=6, choices=range(1, 10), metavar="1-9")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="file of KEY=VALUE credentials (default: src/.env, then ./.env). "
        "Set HF_TOKEN there to lift Hub rate limits.",
    )
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    env_path, env_keys = load_env_file(args.env_file)
    if env_keys:
        print(f"Loaded {', '.join(sorted(env_keys))} from {env_path}", file=sys.stderr)
    if not os.environ.get("HF_TOKEN"):
        print("No HF_TOKEN set; Hub requests will be rate limited.", file=sys.stderr)

    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"error: {out_dir} is not empty (pass --force to overwrite)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior output rather than writing over it: a shorter run leaves
    # higher-numbered shards behind, and Stage 2 globs the directory, so those
    # orphans would be read back as corpus while missing from the manifest.
    for stale in [*out_dir.glob("raw-*.jsonl.gz"), out_dir / "manifest.json"]:
        stale.unlink(missing_ok=True)

    selected = [SOURCES_BY_NAME[n] for n in args.sources]
    weight_sum = sum(s.weight for s in selected)
    total_bytes = int(args.total_mb * 1e6)
    budgets = {s.name: int(total_bytes * s.weight / weight_sum) for s in selected}

    print(f"Collecting {args.total_mb:,.0f} MB into {out_dir}", file=sys.stderr)
    for s in selected:
        print(
            f"  {s.name:<12} {s.domain:<14} {budgets[s.name] / 1e6:>8,.0f} MB  {s.hf_id}",
            file=sys.stderr,
        )

    progress, close_progress, log = make_progress(sum(budgets.values()))
    writer = ShardWriter(out_dir, "raw", int(args.shard_mb * 1e6), args.gzip_level)
    started = datetime.now(timezone.utc)
    all_stats: list[SourceStats] = []

    try:
        for src in selected:
            log(f"-> {src.name} ({src.hf_id} :: {src.config})")
            all_stats.append(
                collect_source(
                    src,
                    writer,
                    budgets[src.name],
                    args.seed,
                    args.max_retries,
                    progress,
                    log,
                )
            )
    finally:
        writer.close()
        close_progress()

    manifest = build_manifest(args, started, all_stats, writer)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print_summary(manifest)
    return 0 if manifest["totals"]["text_bytes"] > 0 else 1


def build_manifest(
    args: argparse.Namespace,
    started: datetime,
    stats: list[SourceStats],
    writer: ShardWriter,
) -> dict[str, Any]:
    """Everything needed to justify and reproduce this corpus in the report."""
    try:
        import datasets as _datasets

        datasets_version = _datasets.__version__
    except Exception:
        datasets_version = None

    return {
        "stage": "01-collection",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "env_file"
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "datasets": datasets_version,
            "hf_authenticated": bool(os.environ.get("HF_TOKEN")),
        },
        "sources": [vars(s) for s in stats],
        "shards": writer.shards,
        "totals": {
            "documents": sum(s.documents for s in stats),
            "text_bytes": sum(s.text_bytes for s in stats),
            "shards": len(writer.shards),
            "compressed_bytes": sum(sh["compressed_bytes"] for sh in writer.shards),
        },
    }


def print_summary(manifest: dict[str, Any]) -> None:
    print("\n" + "=" * 76)
    print(
        f"{'source':<13}{'domain':<15}{'docs':>10}{'MB':>10}"
        f"{'mean doc':>11}{'shards':>9}{'sec':>8}"
    )
    print("-" * 76)
    for s in manifest["sources"]:
        mean = s["text_bytes"] / s["documents"] if s["documents"] else 0
        shards = f"{s['shards_used']}/{s['shards_total']}"
        print(
            f"{s['source']:<13}{s['domain']:<15}{s['documents']:>10,}"
            f"{s['text_bytes'] / 1e6:>10,.1f}{mean:>10,.0f}B{shards:>9}{s['seconds']:>8,.0f}"
        )
    t = manifest["totals"]
    print("-" * 76)
    print(f"{'TOTAL':<28}{t['documents']:>10,}{t['text_bytes'] / 1e6:>10,.1f}")
    print(
        f"{t['shards']} shard(s), {t['compressed_bytes'] / 1e6:,.1f} MB on disk "
        f"(gzip {t['compressed_bytes'] / max(t['text_bytes'], 1):.0%})"
    )
    for s in manifest["sources"]:
        for err in s["errors"]:
            print(f"  ! {s['source']}: {err}", file=sys.stderr)
    if t["text_bytes"] < 1e9:
        print(
            f"\nNOTE: {t['text_bytes'] / 1e9:.2f} GB collected; the assignment "
            "requires >= 1 GB after cleaning. Raise --total-mb.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    code = main()
    # `datasets` leaves an aiohttp event loop behind whose teardown races with
    # interpreter finalization and intermittently aborts the process ("Fatal
    # Python error: PyGILState_Release") *after* the corpus and manifest are
    # safely on disk -- which corrupts the exit code for anything scripting this.
    # Nothing durable is pending here, so leave without running finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
