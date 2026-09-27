"""
Assignment 1, Stage 2: cleaning, normalization and deduplication.

Reads the raw shards written by Stage 1 and emits a cleaned corpus in the same
sharded gzipped JSONL format, plus a manifest recording exactly how many
documents and bytes each filter removed.

Design notes
------------
* The filter chain is ordered *cheapest-first*: a document that a hash lookup
  can reject should never pay for normalization. Dedup on the raw sha256 that
  Stage 1 already computed is therefore the first gate, not the last.
* Every filter reports ``docs`` and ``bytes`` removed. The assignment asks for
  >= 1 GB *after* cleaning, so "how much did each rule cost us" is a number the
  report needs, not a debug aid.
* Dedup keys are stored as 64-bit prefixes rather than 64-char hex digests. For
  the ~560k documents here the birthday collision probability is ~1.7e-8, which
  is far below the rate at which near-duplicates slip past exact matching
  anyway, and it keeps the two membership sets at ~40 MB instead of ~130 MB.
* Only *exact* duplicates are removed (raw, then post-normalization). Near-
  duplicate detection via MinHash/LSH would catch more, but it needs an extra
  dependency and a second pass over the corpus; see the report's limitations
  section.

Usage
-----
    python src/data_cleaning.py --in-dir data/raw --out-dir data/clean
    python src/data_cleaning.py --in-dir data/raw --out-dir data/clean --limit 5000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from data_collection import ShardWriter, make_progress

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# Compiled once at import: these run against every one of ~560k documents, and
# re.sub's internal cache is not large enough to hold them all reliably.

# Script/style blocks must die *with* their contents; a bare tag strip would
# leave the javascript body behind as if it were prose.
RE_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>]{1,2000}>")
# Wikipedia/academic reference markers: [1], [12], [citation needed], [nb 3].
RE_REF_MARKER = re.compile(r"\[\s*(?:\d{1,4}|citation needed|nb\s*\d{1,3}|edit)\s*\]", re.IGNORECASE)
# Markdown emphasis/heading/rule syntax, keeping the words it wraps.
RE_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
RE_MD_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
# Bounded (`.{0,400}?` not `.*?`): an unbounded lazy quantifier between two
# backreferenced delimiters backtracks quadratically on documents full of
# stray asterisks, and web text has plenty of those.
RE_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|`{1,3})(\S.{0,400}?\S|\S)\1", re.DOTALL)
RE_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
# Control characters and zero-width marks, which survive NFKC and corrupt
# downstream token counts without ever being visible.
_INVISIBLE_CODEPOINTS = (
    *range(0x00, 0x09),  # C0 controls, minus tab
    0x0B,
    0x0C,
    *range(0x0E, 0x20),
    0x7F,  # DEL
    *range(0x200B, 0x2010),  # zero-width space .. right-to-left mark
    0x2028,  # line separator
    0x2029,  # paragraph separator
    0xFEFF,  # byte-order mark
)
RE_CONTROL = re.compile("[" + re.escape("".join(map(chr, _INVISIBLE_CODEPOINTS))) + "]")
# Positive class beats the negated `[^\S\n]`: NFKC has already folded exotic
# spaces (NBSP included) down to U+0020 by the time this runs.
RE_SPACES = re.compile(r"[ \t\r\f\v]+")
RE_BLANK_LINES = re.compile(r"\n{3,}")
RE_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")


def normalize(text: str) -> str:
    """Apply the assignment's normalization rules, in dependency order.

    Order matters: markup is stripped before whitespace is collapsed (removing a
    tag leaves a gap), and lowercasing happens after NFKC so that the ligature
    and full-width folds NFKC performs are themselves lowercased.
    """
    text = unicodedata.normalize("NFKC", text)
    # Guards: `"<" in text` is a single C-level scan, while running an unmatched
    # regex over a 3 KB document is not. Most encyclopedic text contains no
    # markup at all, so these skip the majority of the work below.
    if "<" in text:
        text = RE_SCRIPT_STYLE.sub(" ", text)
        text = RE_HTML_COMMENT.sub(" ", text)
        text = RE_HTML_TAG.sub(" ", text)
    if "[" in text:
        text = RE_MD_LINK.sub(r"\1", text)  # keep link text, drop the target
        text = RE_REF_MARKER.sub(" ", text)
    if "://" in text or "www." in text:
        text = RE_URL.sub(" ", text)
    if "#" in text:
        text = RE_MD_HEADING.sub("", text)
    if "*" in text or "_" in text or "`" in text:
        text = RE_MD_RULE.sub("", text)
        text = RE_MD_EMPHASIS.sub(r"\2", text)
    text = text.lower()
    text = RE_CONTROL.sub(" ", text)
    text = RE_SPACES.sub(" ", text)
    text = RE_BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityLimits:
    min_words: int = 50
    min_mean_word_len: float = 2.0  # rejects "a a a a a ..." spam
    max_mean_word_len: float = 15.0  # rejects base64 blobs and concatenated junk
    min_word_char_ratio: float = 0.60  # rejects symbol/markup soup
    max_repeat_ratio: float = 0.30  # one token making up >30% of the document


def quality_verdict(text: str, limits: QualityLimits) -> str | None:
    """Return the name of the first filter that rejects `text`, else None.

    Returning the *reason* rather than a bool is what lets the manifest attribute
    removals to individual rules, which is the evidence the report needs.
    """
    words = RE_WORD.findall(text)
    n_words = len(words)
    if n_words < limits.min_words:
        return "short"

    word_chars = sum(map(len, words))
    mean_len = word_chars / n_words
    if not (limits.min_mean_word_len <= mean_len <= limits.max_mean_word_len):
        return "word_length"

    # Word-character density, reusing the word list. Counting alphanumerics
    # with a separate `findall` allocated one string per character and cost
    # more than the entire normalization pipeline that precedes this.
    if word_chars / len(text) < limits.min_word_char_ratio:
        return "symbol_ratio"

    # Cheap repetition proxy: boilerplate and scraper artifacts collapse to a
    # handful of distinct tokens repeated hundreds of times.
    if Counter(words).most_common(1)[0][1] / n_words > limits.max_repeat_ratio:
        return "repetitive"

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

FILTERS = ("dup_raw", "dup_normalized", "empty", "short", "word_length", "symbol_ratio", "repetitive")


@dataclass
class CleanStats:
    docs_in: int = 0
    docs_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    removed_docs: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FILTERS})
    removed_bytes: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FILTERS})
    per_source_in: dict[str, int] = field(default_factory=dict)
    per_source_out: dict[str, int] = field(default_factory=dict)
    per_source_bytes_out: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def reject(self, reason: str, n_bytes: int) -> None:
        self.removed_docs[reason] += 1
        self.removed_bytes[reason] += n_bytes


def iter_raw_records(in_dir: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    """Stream every record from the Stage 1 shards, in shard order."""
    shards = sorted(in_dir.glob("raw-*.jsonl.gz"))
    if not shards:
        raise SystemExit(f"error: no raw-*.jsonl.gz found in {in_dir}; run data_collection.py first")
    seen = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _key(digest: bytes | str) -> int:
    """64-bit dedup key. See the module docstring on collision probability."""
    if isinstance(digest, str):
        return int(digest[:16], 16)
    return int.from_bytes(digest[:8], "big")


def clean_corpus(
    in_dir: Path,
    writer: ShardWriter,
    limits: QualityLimits,
    limit: int | None,
    progress,
) -> CleanStats:
    stats = CleanStats()
    started = time.monotonic()
    seen_raw: set[int] = set()
    seen_norm: set[int] = set()

    for rec in iter_raw_records(in_dir, limit):
        raw_bytes = rec["n_bytes"]
        source = rec["source"]
        stats.docs_in += 1
        stats.bytes_in += raw_bytes
        stats.per_source_in[source] = stats.per_source_in.get(source, 0) + 1
        progress(raw_bytes)

        # 1. Exact duplicate of a raw document. Free: Stage 1 hashed it already.
        raw_key = _key(rec["sha256"])
        if raw_key in seen_raw:
            stats.reject("dup_raw", raw_bytes)
            continue
        seen_raw.add(raw_key)

        # 2. Normalize, then re-check: two documents differing only in casing,
        #    markup or whitespace are duplicates for a language model even
        #    though their raw hashes differ.
        text = normalize(rec["text"])
        if not text:
            stats.reject("empty", raw_bytes)
            continue

        norm_key = _key(hashlib.sha256(text.encode("utf-8")).digest())
        if norm_key in seen_norm:
            stats.reject("dup_normalized", raw_bytes)
            continue
        seen_norm.add(norm_key)

        # 3. Quality gates.
        reason = quality_verdict(text, limits)
        if reason is not None:
            stats.reject(reason, raw_bytes)
            continue

        out_bytes = len(text.encode("utf-8"))
        writer.write(
            {
                "id": rec["id"],
                "source": source,
                "domain": rec["domain"],
                "url": rec.get("url"),
                "title": rec.get("title"),
                "n_bytes": out_bytes,
                "text": text,
            }
        )
        stats.docs_out += 1
        stats.bytes_out += out_bytes
        stats.per_source_out[source] = stats.per_source_out.get(source, 0) + 1
        stats.per_source_bytes_out[source] = stats.per_source_bytes_out.get(source, 0) + out_bytes

    stats.seconds = round(time.monotonic() - started, 1)
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean, normalize and deduplicate the raw corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-dir", type=Path, default=Path("data/raw"), help="Stage 1 output")
    p.add_argument("--out-dir", type=Path, default=Path("data/clean"), help="cleaned output")
    p.add_argument("--shard-mb", type=float, default=128.0, help="uncompressed JSONL per shard")
    p.add_argument("--gzip-level", type=int, default=6, choices=range(1, 10), metavar="1-9")
    p.add_argument("--min-words", type=int, default=50, help="drop documents below this word count")
    p.add_argument("--limit", type=int, default=None, help="stop after N raw docs (smoke tests)")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"error: {out_dir} is not empty (pass --force to overwrite)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("clean-*.jsonl.gz"), out_dir / "manifest.json"]:
        stale.unlink(missing_ok=True)

    raw_manifest_path = in_dir / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text()) if raw_manifest_path.is_file() else {}
    total_in = raw_manifest.get("totals", {}).get("text_bytes", 0)

    print(f"Cleaning {in_dir} -> {out_dir}", file=sys.stderr)
    limits = QualityLimits(min_words=args.min_words)
    progress, close_progress, _log = make_progress(total_in)
    writer = ShardWriter(out_dir, "clean", int(args.shard_mb * 1e6), args.gzip_level)
    started = datetime.now(timezone.utc)

    try:
        stats = clean_corpus(in_dir, writer, limits, args.limit, progress)
    finally:
        writer.close()
        close_progress()

    manifest = build_manifest(args, started, stats, limits, writer, raw_manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print_summary(manifest)
    return 0 if stats.docs_out > 0 else 1


def build_manifest(
    args: argparse.Namespace,
    started: datetime,
    stats: CleanStats,
    limits: QualityLimits,
    writer: ShardWriter,
    raw_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "02-cleaning",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "quality_limits": vars(limits),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "input": {
            "stage": raw_manifest.get("stage"),
            "documents": stats.docs_in,
            "text_bytes": stats.bytes_in,
        },
        "removed": {
            f: {"documents": stats.removed_docs[f], "input_bytes": stats.removed_bytes[f]}
            for f in FILTERS
        },
        "per_source": {
            s: {
                "documents_in": stats.per_source_in.get(s, 0),
                "documents_out": stats.per_source_out.get(s, 0),
                "text_bytes_out": stats.per_source_bytes_out.get(s, 0),
            }
            for s in sorted(stats.per_source_in)
        },
        "shards": writer.shards,
        "totals": {
            "documents": stats.docs_out,
            "text_bytes": stats.bytes_out,
            "shards": len(writer.shards),
            "compressed_bytes": sum(sh["compressed_bytes"] for sh in writer.shards),
            "seconds": stats.seconds,
        },
    }


def print_summary(manifest: dict[str, Any]) -> None:
    inp, tot = manifest["input"], manifest["totals"]
    print("\n" + "=" * 68)
    print(f"{'filter':<20}{'docs removed':>16}{'% of input docs':>18}{'MB removed':>14}")
    print("-" * 68)
    for name, r in manifest["removed"].items():
        if not r["documents"]:
            continue
        pct = 100 * r["documents"] / inp["documents"] if inp["documents"] else 0
        print(f"{name:<20}{r['documents']:>16,}{pct:>17.2f}%{r['input_bytes'] / 1e6:>14,.1f}")
    print("-" * 68)
    print(f"{'source':<20}{'docs out':>16}{'MB out':>18}")
    for src, s in manifest["per_source"].items():
        print(f"{src:<20}{s['documents_out']:>16,}{s['text_bytes_out'] / 1e6:>18,.1f}")
    print("-" * 68)
    kept = 100 * tot["text_bytes"] / inp["text_bytes"] if inp["text_bytes"] else 0
    print(
        f"in  {inp['documents']:>9,} docs  {inp['text_bytes'] / 1e9:>6.2f} GB\n"
        f"out {tot['documents']:>9,} docs  {tot['text_bytes'] / 1e9:>6.2f} GB "
        f"({kept:.1f}% of input bytes, {tot['shards']} shard(s), "
        f"{tot['compressed_bytes'] / 1e6:,.0f} MB on disk) in {tot['seconds']:,.0f}s"
    )
    if tot["text_bytes"] < 1e9:
        print(
            f"\nNOTE: {tot['text_bytes'] / 1e9:.2f} GB after cleaning; the assignment "
            "requires >= 1 GB. Re-run Stage 1 with a larger --total-mb.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
