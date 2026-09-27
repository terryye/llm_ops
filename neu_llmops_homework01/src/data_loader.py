"""
Assignment 1, Stage 4: custom PyTorch data loaders.

Three dataset classes over the Stage 3 token files, covering the two access
patterns a pretraining run actually needs plus the padded variant the
assignment asks to see:

``PackedBlockDataset``     map-style, memory-mapped, random access. The default.
``StreamingBlockDataset``  ``IterableDataset`` for corpora larger than RAM.
``DocumentDataset``        variable-length documents + a padding collate, to
                           show the alternative to packing and measure its cost.

Design notes
------------
* **Memory mapping, not loading.** The token files are read with
  ``np.memmap``, so a 1 GB+ corpus costs a page-cache mapping rather than
  resident memory, and a batch materialises only the rows it touches.
* **The memmap is opened lazily, per worker.** A ``np.memmap`` created in the
  parent process and inherited through ``fork`` shares a file offset and does
  not survive pickling to spawned workers. Opening on first access inside the
  worker is what makes ``num_workers > 0`` safe here.
* **Causal-LM targets.** Each item reads ``block_size + 1`` tokens and returns
  ``input_ids = t[:-1]`` with ``labels = t[1:]``, so the loader emits training
  pairs rather than raw blocks and needs no shifting in the training step.
* **Packed blocks need no attention mask** -- every position is a real token.
  The mask only appears on the padded path, where it carries information.

Usage
-----
    python src/data_loader.py --tokens-dir data/tokens --sample-out sample_dataset.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

TOKEN_DTYPE = np.uint16  # must match tokenize_corpus.TOKEN_DTYPE


def load_token_manifest(tokens_dir: Path) -> dict[str, Any]:
    path = tokens_dir / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run tokenize_corpus.py first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Map-style: random access over packed blocks
# ---------------------------------------------------------------------------


class PackedBlockDataset(Dataset):
    """Fixed-size blocks of packed tokens, addressed by index.

    Pairs with ``DataLoader(shuffle=True)``: because every block is the same
    length and lives at a computable offset, shuffling is an index permutation
    and costs nothing on top of the read.
    """

    def __init__(self, path: Path, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        if not self.path.is_file():
            raise SystemExit(f"error: {self.path} not found; run tokenize_corpus.py first")
        n_tokens = self.path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
        # Each item needs block_size + 1 tokens (inputs plus the shifted target),
        # so the final block is only usable if that extra token exists.
        self.n_tokens = n_tokens
        self.n_blocks = max(0, (n_tokens - 1) // block_size)
        self._tokens: np.memmap | None = None

    def _memmap(self) -> np.memmap:
        # Lazy so that each DataLoader worker gets its own mapping; see the
        # module docstring.
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.n_blocks
        if not 0 <= idx < self.n_blocks:
            raise IndexError(f"block {idx} out of range for {self.n_blocks} blocks")
        start = idx * self.block_size
        # int64 is what nn.Embedding and cross_entropy expect; the widening
        # happens here, on one block, rather than on the whole corpus.
        chunk = np.asarray(self._memmap()[start : start + self.block_size + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }


# ---------------------------------------------------------------------------
# Iterable: streaming for corpora that do not fit in RAM
# ---------------------------------------------------------------------------


class StreamingBlockDataset(IterableDataset):
    """Streams blocks in shard order with a bounded shuffle buffer.

    Use when the token file is too large to index eagerly or lives behind a
    network filesystem. Blocks are split across workers by stride so that no
    two workers emit the same block, and ``shuffle_buffer`` bounds the memory
    cost of decorrelating block order at ``buffer * block_size * 8`` bytes.
    """

    def __init__(self, path: Path, block_size: int, shuffle_buffer: int = 1024, seed: int = 0):
        self.path = Path(path)
        self.block_size = block_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0
        n_tokens = self.path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
        self.n_blocks = max(0, (n_tokens - 1) // block_size)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle between epochs; without this every epoch has one order."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_blocks

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        start, stride = (0, 1) if worker is None else (worker.id, worker.num_workers)
        tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        rng = random.Random(hash((self.seed, self.epoch, start)))

        buffer: list[dict[str, torch.Tensor]] = []
        for idx in range(start, self.n_blocks, stride):
            offset = idx * self.block_size
            chunk = np.asarray(tokens[offset : offset + self.block_size + 1], dtype=np.int64)
            item = {
                "input_ids": torch.from_numpy(chunk[:-1]),
                "labels": torch.from_numpy(chunk[1:]),
            }
            if self.shuffle_buffer <= 1:
                yield item
                continue
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                # Swap-and-pop: yields a random element in O(1) without the
                # O(n) shift that list.pop(i) would cost on every block.
                j = rng.randrange(len(buffer))
                buffer[j], buffer[-1] = buffer[-1], buffer[j]
                yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer


# ---------------------------------------------------------------------------
# Variable-length documents + padding (the alternative to packing)
# ---------------------------------------------------------------------------


class DocumentDataset(Dataset):
    """One un-packed document per item, truncated to ``max_length``.

    Kept to satisfy the assignment's "handle variable-length sequences with
    padding or truncation" requirement and to quantify what packing saves:
    ``pad_collate`` reports how many positions in each batch are padding.
    """

    def __init__(self, clean_dir: Path, tokenizer, max_length: int, limit: int = 2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts: list[str] = []
        for shard in sorted(Path(clean_dir).glob("clean-*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    self.texts.append(json.loads(line)["text"])
                    if len(self.texts) >= limit:
                        return
            if len(self.texts) >= limit:
                return

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def make_pad_collate(pad_token_id: int):
    """Right-pad a batch to its longest member and mask the padding out.

    ``labels`` uses -100 for pad positions, the ignore index
    ``torch.nn.functional.cross_entropy`` skips, so padding never contributes
    a gradient.
    """

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        lengths = [len(b["input_ids"]) for b in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(ids)] = ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pad_fraction": torch.tensor(1.0 - sum(lengths) / (len(batch) * width)),
        }

    return collate


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_dataloader(
    tokens_dir: Path,
    split: str = "train",
    batch_size: int = 8,
    block_size: int | None = None,
    shuffle: bool = True,
    streaming: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Build the loader for one split, reading block_size from the manifest."""
    manifest = load_token_manifest(tokens_dir)
    block_size = block_size or manifest["format"]["block_size"]
    path = tokens_dir / manifest["splits"][split]["path"]

    if streaming:
        dataset: Dataset | IterableDataset = StreamingBlockDataset(
            path, block_size, shuffle_buffer=1024 if shuffle else 1, seed=seed
        )
        shuffle = False  # an IterableDataset shuffles itself; DataLoader must not
    else:
        dataset = PackedBlockDataset(path, block_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # keeps every batch the same shape
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


# ---------------------------------------------------------------------------
# Entry point: demonstrate the loaders and write the sample deliverable
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exercise the data loaders and save a sample batch file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    p.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    p.add_argument("--sample-out", type=Path, default=Path("sample_dataset.pt"))
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="measured loader statistics, consumed by make_report.py "
        "(default: loader_metrics.json inside --tokens-dir)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sample-blocks", type=int, default=10, help="blocks to save (assignment: 5-10)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260918)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Default it here, not in argparse, so it follows --tokens-dir. A fixed
    # default meant a run against one token directory overwrote the metrics of
    # another, and the report silently picked up the wrong numbers.
    if args.metrics_out is None:
        args.metrics_out = args.tokens_dir / "loader_metrics.json"
    torch.manual_seed(args.seed)
    manifest = load_token_manifest(args.tokens_dir)
    block_size = manifest["format"]["block_size"]
    tok_id = manifest["tokenizer"]["id"]

    print("=" * 68)
    print(f"tokens from {args.tokens_dir}  (block_size {block_size}, tokenizer {tok_id})")
    print("=" * 68)

    # 1. Map-style packed loader -------------------------------------------
    train = make_dataloader(
        args.tokens_dir, "train", args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )
    batch = next(iter(train))
    print(f"\n[1] PackedBlockDataset (map-style, shuffled, {args.num_workers} workers)")
    print(f"    blocks           {len(train.dataset):,}")
    print(f"    batches/epoch    {len(train):,}")
    print(f"    input_ids        {tuple(batch['input_ids'].shape)}  {batch['input_ids'].dtype}")
    print(f"    labels           {tuple(batch['labels'].shape)}  {batch['labels'].dtype}")
    # The whole point of the shift: labels lead inputs by exactly one position.
    assert torch.equal(batch["input_ids"][0, 1:], batch["labels"][0, :-1])
    print("    labels == inputs shifted by one: OK")

    # 2. Streaming loader ---------------------------------------------------
    stream = make_dataloader(
        args.tokens_dir, "train", args.batch_size, streaming=True,
        shuffle=True, num_workers=args.num_workers, seed=args.seed,
    )
    s_batch = next(iter(stream))
    print("\n[2] StreamingBlockDataset (IterableDataset, worker-sharded)")
    print(f"    input_ids        {tuple(s_batch['input_ids'].shape)}")
    print("    memory           bounded by shuffle buffer, not corpus size")

    # 3. Padded variable-length loader --------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    tokenizer.model_max_length = int(1e9)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    docs = DocumentDataset(args.clean_dir, tokenizer, max_length=block_size, limit=512)
    pad_loader = DataLoader(
        docs, batch_size=args.batch_size, shuffle=True, collate_fn=make_pad_collate(pad_id)
    )
    waste = [b["pad_fraction"].item() for b in pad_loader]
    p_batch = next(iter(pad_loader))
    print("\n[3] DocumentDataset + pad_collate (the alternative to packing)")
    print(f"    input_ids        {tuple(p_batch['input_ids'].shape)} (ragged, padded per batch)")
    print(f"    attention_mask   {tuple(p_batch['attention_mask'].shape)}")
    print(f"    mean padding     {100 * sum(waste) / len(waste):.1f}% of positions wasted")
    print("    packing wastes   0.0% -- every position in [1] is a real token")

    # 4. Sample deliverable --------------------------------------------------
    sample_loader = make_dataloader(
        args.tokens_dir, "train", args.sample_blocks, shuffle=False, num_workers=0
    )
    sample = next(iter(sample_loader))
    payload = {
        "input_ids": sample["input_ids"],
        "labels": sample["labels"],
        "block_size": block_size,
        "n_blocks": args.sample_blocks,
        "tokenizer": tok_id,
        "vocab_size": manifest["tokenizer"]["vocab_size"],
        "dtype": "int64 (from uint16 on disk)",
        "provenance": {
            "stage3_manifest": str(args.tokens_dir / "manifest.json"),
            "train_tokens": manifest["splits"]["train"]["tokens"],
            "val_tokens": manifest["splits"]["val"]["tokens"],
        },
    }
    torch.save(payload, args.sample_out)
    size_kb = args.sample_out.stat().st_size / 1024
    print(f"\n[4] Saved {args.sample_out} "
          f"({args.sample_blocks} blocks, {tuple(sample['input_ids'].shape)}, {size_kb:,.0f} KB)")

    decoded = tokenizer.decode(sample["input_ids"][0][:60])
    print(f"\n    first block decodes to:\n    {decoded[:220]!r}")

    # Measured here rather than quoted in prose: the report is generated from
    # this file, so its numbers cannot drift from what the code actually does.
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "block_size": block_size,
                "packed_blocks": len(train.dataset),
                "batches_per_epoch": len(train),
                "packed_batch_shape": list(batch["input_ids"].shape),
                "packed_pad_fraction": 0.0,
                "padded_batch_shape": list(p_batch["input_ids"].shape),
                "padded_mean_pad_fraction": sum(waste) / len(waste),
                "padded_batches_measured": len(waste),
                "num_workers": args.num_workers,
                "sample_file": str(args.sample_out),
                "sample_shape": list(sample["input_ids"].shape),
                "sample_bytes": args.sample_out.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"    metrics written to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
