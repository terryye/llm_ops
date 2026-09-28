"""Correctness checks for the model, data windows, training loop and checkpoints.

Run with:  python -m pytest -q tests
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import PAGE, TokenWindowDataset  # noqa: E402
from model import GPTConfig, MiniGPT  # noqa: E402
from train import TrainConfig, load_checkpoint, lr_at, read_metrics, train  # noqa: E402

TINY = GPTConfig(vocab_size=97, block_size=16, n_layer=2, n_embd=32, n_head=4)


def test_logits_shape_and_initial_loss():
    torch.manual_seed(0)
    model = MiniGPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (3, 16))
    y = torch.randint(0, TINY.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, TINY.vocab_size)
    # A freshly initialized model should be close to uniform over the vocabulary.
    # (Targets must be independent of the inputs: with a tied head, each
    # position's own input token gets a raised logit even before training.)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.3


def test_attention_is_causal():
    """Changing a future token must not change any earlier position's logits."""
    torch.manual_seed(0)
    model = MiniGPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % TINY.vocab_size
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_head_is_tied_and_counted_once():
    model = MiniGPT(TINY)
    assert model.lm_head.weight is model.tok_emb.weight
    unique = sum(p.numel() for p in model.parameters())  # parameters() dedups shared tensors
    assert model.num_parameters()["total"] == unique


def test_model_can_overfit_one_batch():
    """End-to-end gradient sanity check: loss must collapse on a single batch."""
    torch.manual_seed(0)
    model = MiniGPT(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, TINY.vocab_size, (4, 17))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_micro_batch_gradients_match_full_batch():
    torch.manual_seed(0)
    model = MiniGPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (8, 17))
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for chunk in x.split(2):
        _, loss = model(chunk[:, :-1], chunk[:, 1:])
        (loss * chunk.size(0) / x.size(0)).backward()
    for g_full, p in zip(full, model.parameters()):
        assert torch.allclose(g_full, p.grad, atol=1e-6)


def test_lr_schedule_warms_up_then_decays_to_floor():
    cfg = TrainConfig(lr=1e-3, warmup_frac=0.1, min_lr_ratio=0.1)
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] < lrs[5] < lrs[9] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))
    assert lrs[-1] == pytest.approx(1e-4, rel=0.01)


@pytest.fixture()
def token_dir(tmp_path: Path) -> Path:
    """A miniature Assignment 1 output: 40 train pages, 6 val pages."""
    rng = np.random.default_rng(0)
    for split, pages in (("train", 40), ("val", 6)):
        rng.integers(0, 97, size=pages * PAGE, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": {"dtype": "uint16"},
        "tokenizer": {"id": "synthetic", "vocab_size": 97},
    }))
    return tmp_path


def test_windows_are_shifted_by_one_and_cover_the_same_pages(token_dir: Path):
    raw = np.fromfile(token_dir / "train.bin", dtype=np.uint16)
    a = TokenWindowDataset(token_dir / "train.bin", 32, max_tokens=10 * PAGE, seed=3)
    b = TokenWindowDataset(token_dir / "train.bin", 128, max_tokens=10 * PAGE, seed=3)
    assert np.array_equal(a.pages, b.pages)  # sequence length does not change the data
    assert len(a) == 10 * PAGE // 32 and a.n_tokens == b.n_tokens == 10 * PAGE
    x, y = a[5]
    start = int(a.pages[0]) * PAGE + 5 * 32
    assert torch.equal(x, torch.from_numpy(raw[start : start + 32].astype(np.int64)))
    assert torch.equal(y[:-1], x[1:])


def test_train_resume_and_checkpoint_roundtrip(token_dir: Path, tmp_path: Path):
    cfg = TrainConfig(
        name="tiny", out_dir=str(tmp_path / "runs"), tokens_dir=str(token_dir),
        n_layer=1, n_embd=32, n_head=2, seq_len=32, train_tokens=8 * PAGE, val_tokens=2 * PAGE,
        epochs=2, batch_size=16, micro_batch_size=8, evals_per_epoch=2, log_every=4,
        num_workers=0, device="cpu",
    )
    export = tmp_path / "tiny.pt"
    summary = train(cfg, export=export)
    records = read_metrics(cfg.run_dir)
    epochs = [r for r in records if r["kind"] == "epoch"]
    assert [r["epoch"] for r in epochs] == [1, 2]
    assert summary["steps"] == 2 * (8 * PAGE // 32 // 16)
    assert math.isclose(summary["final_val_ppl"], math.exp(summary["final_val_loss"]))

    model, payload = load_checkpoint(export)
    assert payload["summary"]["final_val_loss"] == summary["final_val_loss"]
    with torch.no_grad():
        out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)  # longer than block_size: context is cropped

    # Resuming a finished run re-evaluates without duplicating any log record.
    again = train(cfg, resume=True)
    assert again["final_val_loss"] == pytest.approx(summary["final_val_loss"], abs=1e-6)
    assert len(read_metrics(cfg.run_dir)) == len(records)
