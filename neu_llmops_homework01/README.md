# Assignment 1: Data Collection and Preprocessing for Foundation Model Pre-Training

A four-stage pipeline that collects a multi-domain raw text corpus from public
Hugging Face datasets, cleans and deduplicates it, tokenizes it into packed
GPT-2 blocks, and serves it through custom PyTorch data loaders.

## Quick start

```bash
pip install -r requirements.txt

# CPU-only torch is enough here and saves ~2 GB over the CUDA build:
pip install --index-url https://download.pytorch.org/whl/cpu torch

python data_collection_preprocessing.py            # full run
python data_collection_preprocessing.py --smoke    # ~2 min end-to-end check
```

An `HF_TOKEN` is not required (all sources are public) but avoids Hub rate
limits. Put it in `src/.env` as `HF_TOKEN=hf_...`.

## Pipeline

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Collection | `src/data_collection.py` | Hugging Face Hub | `data/raw/raw-*.jsonl.gz` |
| 2. Cleaning | `src/data_cleaning.py` | `data/raw` | `data/clean/clean-*.jsonl.gz` |
| 3. Tokenization | `src/tokenize_corpus.py` | `data/clean` | `data/tokens/{train,val}.bin` |
| 4. Loaders | `src/data_loader.py` | `data/tokens` | `sample_dataset.pt` |
| Report | `src/make_report.py` | all manifests | `Assignment1_Report.pdf` |

Every stage runs standalone and writes a `manifest.json` next to its output
recording inputs, arguments, environment, per-shard SHA-256 digests and
statistics. Re-run any single stage without repeating the ones before it:

```bash
python data_collection_preprocessing.py --skip collect clean
```

## Using the loaders

```python
import sys; sys.path.insert(0, "src")
from data_loader import make_dataloader

train = make_dataloader("data/tokens", split="train", batch_size=8, shuffle=True)
for batch in train:
    batch["input_ids"]  # (8, 1024) int64
    batch["labels"]     # (8, 1024) int64, inputs shifted by one
```

Pass `streaming=True` for the `IterableDataset` variant, which bounds memory by
a shuffle buffer rather than the corpus size.

## Design decisions

- **Stratified shard sampling.** Dataset shards are ordered non-randomly
  (Wikipedia's alphabetically), so Stage 1 walks them in a seeded random order
  and takes an equal byte quota from each. `IterableDataset.shuffle()` also
  fixes the clustering but buffers Arrow tables rather than rows and opens ten
  shards at once, which exhausted memory on an 8 GB machine. See section 5.1 of
  the report.
- **1.5x raw budget.** The >= 1 GB requirement applies after cleaning, so Stage
  1 collects 1.5 GB to leave room for deduplication and quality filtering.
- **Packing, not padding.** Documents are concatenated with an end-of-text
  separator and cut into fixed blocks, so documents longer than the block size
  span several blocks instead of being truncated, and no batch position is
  wasted on padding.
- **uint16 token storage.** GPT-2's 50,257-token vocabulary fits in 16 bits;
  int64 would quadruple the artifact for no benefit.
- **Hash-based train/val split.** Per document, by SHA-256 of the document id,
  so every domain is represented and the split is reproducible.

## Deliverables

- `data_collection_preprocessing.py` plus the stage scripts in `src/`
- `sample_dataset.pt` (10 blocks of 1024 tokens, saved with `torch.save`)
- `Assignment1_Report.pdf`

## Layout

```
data_collection_preprocessing.py   top-level pipeline runner
src/
  data_collection.py               stage 1
  data_cleaning.py                 stage 2
  tokenize_corpus.py               stage 3
  data_loader.py                   stage 4
  make_report.py                   PDF report, generated from the manifests
data/                              generated; not tracked
sample_dataset.pt                  sample tokenized batches
Assignment1_Report.pdf             report
```
