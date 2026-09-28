"""
Assignment 2: data loading over the Assignment 1 token files.

Assignment 1 produced ``train.bin`` and ``val.bin``: flat, packed GPT-2 token
ids stored as uint16, with an end-of-text token between documents. This module
cuts them into the short training windows this assignment asks for
(32-128 tokens) and serves them as shuffled batches.

Design notes
------------
* **Same tokens at every sequence length.** A run trains on a seeded random
  sample of Assignment 1's 1024-token blocks ("pages"), and each page is cut
  into ``1024 / seq_len`` windows. Because the page sample depends only on the
  seed and the token budget, the seq_len experiments see exactly the same text
  and differ only in how it is windowed. Sampling windows independently per
  length would confound context length with data.
* **Sampled pages, not a prefix.** Assignment 1 wrote documents in shard
  order, so a prefix of train.bin over-represents whichever shards come
  first. Random pages spread the budget across the whole corpus.
* **Memory mapped and opened lazily**, for the same reason as in Assignment 1:
  each DataLoader worker must open its own mapping instead of inheriting the
  parent's. Only the sampled pages are ever read from disk.
* **Targets are the inputs shifted by one.** Each window reads seq_len + 1
  tokens; the extra token is the label for the last position. It may come
  from the next page, which is fine: pages are contiguous in the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_DTYPE = np.uint16  # must match Assignment 1's tokenize_corpus.TOKEN_DTYPE
PAGE = 1024  # Assignment 1 block size; the unit of sampling


def load_token_manifest(tokens_dir: Path) -> dict[str, Any]:
    path = Path(tokens_dir) / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run Assignment 1's pipeline first")
    manifest = json.loads(path.read_text())
    if manifest["format"]["dtype"] != np.dtype(TOKEN_DTYPE).name:
        raise SystemExit(f"error: expected {np.dtype(TOKEN_DTYPE).name} tokens, "
                         f"manifest says {manifest['format']['dtype']}")
    return manifest


class TokenWindowDataset(Dataset):
    """Fixed-length (input, target) windows over a random sample of pages.

    ``max_tokens=None`` uses every page, which is what evaluation wants.
    """

    def __init__(self, path: Path, seq_len: int, max_tokens: int | None = None, seed: int = 0):
        if PAGE % seq_len:
            raise ValueError(f"seq_len={seq_len} must divide the {PAGE}-token page size")
        self.path = Path(path)
        if not self.path.is_file():
            raise SystemExit(f"error: {self.path} not found; run Assignment 1's pipeline first")
        self.seq_len = seq_len
        self.windows_per_page = PAGE // seq_len

        n_tokens = self.path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
        # The last page is only usable if one more token follows it.
        n_pages = (n_tokens - 1) // PAGE
        if max_tokens is None or max_tokens >= n_pages * PAGE:
            pages = np.arange(n_pages, dtype=np.int64)
        else:
            k = max(1, max_tokens // PAGE)
            rng = np.random.default_rng(seed)
            # Sorted so reads walk the file forward; order is irrelevant
            # because the DataLoader shuffles windows anyway.
            pages = np.sort(rng.choice(n_pages, size=k, replace=False))
        self.pages = pages
        self._tokens: np.memmap | None = None

    @property
    def n_tokens(self) -> int:
        """Input tokens covered by one pass over the dataset."""
        return len(self) * self.seq_len

    def _memmap(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return len(self.pages) * self.windows_per_page

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        page, window = divmod(idx, self.windows_per_page)
        start = int(self.pages[page]) * PAGE + window * self.seq_len
        chunk = np.asarray(self._memmap()[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk)
        return x[:-1], x[1:]


def make_loader(
    dataset: TokenWindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """A DataLoader whose shuffle order is a pure function of ``seed``.

    Callers that want a new order per epoch pass ``seed + epoch``, which also
    makes resuming from a checkpoint reproduce the order it would have seen.
    """
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
    )
