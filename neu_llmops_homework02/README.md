# Assignment 2: Building a Small-Scale Foundation Model from Scratch

A mini-GPT (decoder-only transformer, 1-2 layers, embedding size 64-256, 4 heads)
written from PyTorch primitives and trained from scratch, for next-token
prediction, on the tokenized corpus from Assignment 1. It includes a
one-factor-at-a-time hyperparameter sweep, loss and perplexity curves, and a
report generated from the run logs.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # see the file for the right torch build

python -m pytest -q tests                # 8 correctness tests, ~5 s on CPU
python train_mini_gpt.py --smoke         # tiny end-to-end run, ~10 min on CPU
python train_mini_gpt.py                 # full sweep + figures + report (use a GPU)
```

The data comes from Assignment 1: by default the scripts read
`../neu_llmops_homework01/data/tokens/{train,val}.bin` and its `manifest.json`
(pass `--tokens-dir` to `src/experiments.py` or `src/train.py` to point elsewhere). Produce them with
`python data_collection_preprocessing.py` in that directory.

The device is picked automatically: CUDA, then Apple MPS, then CPU. Training
runs in fp32, except for bf16 autocast on GPUs with native bf16 (Ampere and
newer). On the GTX 1650 used here, fp16 autocast was 12% slower than fp32.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1. Sweep | `src/experiments.py` | `runs/<name>/{config.json,metrics.jsonl,checkpoint.pt,summary.json}`, `runs/sweep.json`, `mini_gpt_checkpoint.pt` |
| 2. Figures | `src/plots.py` | `figures/{baseline_curves,sweep_curves,sweep_perplexity}.png` |
| 3. Samples | `src/generate.py` | `samples.json` |
| 4. Report | `src/make_report.py` | `Assignment2_Report.pdf` |

`train_mini_gpt.py` runs all four in order; `--skip train` redraws figures and
the report from existing logs. The sweep skips finished runs and resumes
interrupted ones from their last epoch checkpoint.

## Training a single run

```bash
python src/train.py --name my_run --n-embd 256 --lr 5e-4 --epochs 3
python src/train.py --name my_run --resume                  # continue after a crash
python src/train.py --name bench --benchmark-steps 30        # throughput only
```

Every field of `TrainConfig` in `src/train.py` is a flag (`--batch-size`,
`--seq-len`, `--n-layer`, `--train-tokens`, ...).

## Loading the checkpoint

```python
import sys; sys.path.insert(0, "src")
from train import load_checkpoint

model, payload = load_checkpoint("mini_gpt_checkpoint.pt", device="cpu")
payload["model_config"]   # architecture, used to rebuild the model
payload["summary"]        # final validation loss / perplexity of the run
```

Or sample text from it: `python src/generate.py --prompt "The history of"`.

## Design decisions

- **Hand-written attention.** The QKV projection, head split, scaled scores,
  causal mask, softmax and head merge are explicit in `src/model.py` rather than
  hidden in `nn.MultiheadAttention`. At 128 tokens a fused kernel saves nothing
  measurable, because the vocabulary projection dominates the cost.
- **Tied embedding and output head.** With GPT-2's 50,257-token vocabulary the
  embedding table is about 94% of the baseline's parameters. Tying avoids a
  second table of the same size.
- **Same text at every sequence length.** Runs sample Assignment 1's 1024-token
  blocks and cut them into windows, so the seq_len experiments see identical
  tokens and differ only in context length.
- **Fixed token budget across runs.** Batch-size runs take fewer or more steps
  over the same tokens, which is the trade-off worth measuring.
- **Micro-batching.** Batches are split into micro-batches whose gradients
  accumulate before one optimizer step. The logits for a 64 x 128 batch are
  1.6 GB in fp32, and the loss and backward pass need several times that, so
  a 4 GB GPU runs micro-batches of 16 sequences.
- **Token-weighted perplexity.** exp(mean cross-entropy over every validation
  token), not a mean of per-batch perplexities.
- **Resumable, append-only logs.** One JSON line per event; resuming drops the
  records past the checkpoint so the curves never double-count a step.

## Deliverables

- Code: `src/model.py` (model), `src/train.py` (training loop, checkpoint
  save/load, loss and perplexity logging), `src/experiments.py` (hyperparameter
  sweep), `src/plots.py`, `src/generate.py`, `src/make_report.py`
- `mini_gpt_checkpoint.pt`: weights, config and final metrics of the best run
- `figures/`: training loss and perplexity curves
- `Assignment2_Report.pdf`

## Layout

```
train_mini_gpt.py         top-level runner
src/
  model.py                MiniGPT: embeddings, causal self-attention, MLP, blocks
  data.py                 windows over the Assignment 1 token files
  train.py                one training run; checkpoint save/load
  experiments.py          baseline + one-factor sweep; exports the best checkpoint
  plots.py                figures from the run logs
  generate.py             load a checkpoint and sample text
  make_report.py          PDF report from the run logs
tests/test_mini_gpt.py    causality, gradients, data windows, resume, checkpoints
runs/                     per-run logs and checkpoints (training state not tracked)
figures/                  loss and perplexity curves
```
