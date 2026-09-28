"""
Assignment 2: train one mini-GPT run.

Each run trains for a fixed number of epochs over a fixed, seeded sample of
the Assignment 1 training tokens, and writes to ``<out-dir>/<name>/``:

    config.json      every setting, plus environment and parameter counts
    metrics.jsonl    one JSON object per line; "kind" is step | eval | epoch
    checkpoint.pt    full training state at the last finished epoch (resumable)
    summary.json     final numbers, written only when the run completes

Design notes
------------
* **Loop.** Forward, cross-entropy on next-token targets, backward, gradient
  clipping, AdamW step, LR schedule step. Batches larger than
  ``--micro-batch-size`` are split into micro-batches whose gradients
  accumulate before the single optimizer step, so the batch size being
  studied is the optimization batch, independent of what fits in GPU memory.
  The logits for one 64 x 128 batch over GPT-2's vocabulary are 1.6 GB in
  fp32, and the log-softmax and both gradients each need as much again, so a
  4 GB card fits micro-batches of 16 sequences.
* **Precision.** fp32 by default; bf16 autocast only on GPUs with native bf16
  (compute capability 8.0+). On a GTX 1650 fp16 autocast measured 12% slower
  than fp32: Turing's TU117 has no tensor cores, so the casts cost more than
  the half-precision math saves.
* **Perplexity is exp(mean per-token cross-entropy)** on held-out data. The
  mean is token-weighted over the whole evaluation set, not an average of
  per-batch perplexities, which would be biased upward by Jensen's inequality.
* **Schedule.** Linear warmup, then cosine decay to 10% of the peak LR.
  Warmup is a fraction of total steps so runs with different batch sizes (and
  therefore different step counts) get the same shape of schedule.
* **Weight decay** applies to matrices only; biases, LayerNorm gains and the
  positional table stay undecayed, as in GPT-2 and nanoGPT.
* **Evaluation points are spaced in tokens seen, not steps**, so curves from
  runs with different batch sizes line up on the same x axis.
* **Reproducibility.** The seed fixes the model initialization, the page
  sample and each epoch's shuffle order (seed + epoch). A resumed run
  therefore sees the same batches as an uninterrupted one. On CUDA, results
  are repeatable to within floating-point nondeterminism of atomics in the
  embedding backward pass, not bit-for-bit.

Usage
-----
    python src/train.py --name baseline
    python src/train.py --name wide --n-embd 256 --lr 5e-4
    python src/train.py --name baseline --resume          # continue after a crash
    python src/train.py --name bench --benchmark-steps 30 # throughput only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TokenWindowDataset, load_token_manifest, make_loader  # noqa: E402
from model import GPTConfig, MiniGPT  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENS_DIR = HERE.parent.parent / "neu_llmops_homework01" / "data" / "tokens"


@dataclass
class TrainConfig:
    name: str = "baseline"
    out_dir: str = "runs"
    tokens_dir: str = str(DEFAULT_TOKENS_DIR)
    # model
    n_layer: int = 2
    n_embd: int = 128
    n_head: int = 4
    dropout: float = 0.0
    # data
    seq_len: int = 128
    train_tokens: int = 1_572_864  # 1,536 pages of 1024 tokens per epoch
    val_tokens: int = 131_072  # periodic evaluation subset
    final_val_tokens: int = 524_288  # final evaluation; 0 means the whole validation split
    epochs: int = 2
    # optimization
    batch_size: int = 32
    micro_batch_size: int = 16  # fits a 4 GB GPU; see the module docstring
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # bookkeeping
    evals_per_epoch: int = 4
    log_every: int = 10
    seed: int = 1337
    device: str = "auto"
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    num_workers: int = 2

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def model_config(self, vocab_size: int) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size, block_size=self.seq_len, n_layer=self.n_layer,
            n_embd=self.n_embd, n_head=self.n_head, dropout=self.dropout,
        )


# ---------------------------------------------------------------------------
# Device and precision
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        # bf16 only where the hardware has it natively (Ampere and newer). It
        # needs no loss scaling, unlike fp16, which is also slower than fp32
        # on cards without tensor cores. CPU and MPS stay in fp32.
        if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0):
            return torch.bfloat16
        return torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)


def environment(device: torch.device) -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env["gpu"] = props.name
        env["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
        env["cuda"] = torch.version.cuda
    elif device.type == "cpu":
        env["cpu_threads"] = torch.get_num_threads()
        # Cores this process may actually run on; a cgroup or an external
        # pinning can make this smaller than the thread count, which shows up
        # as a large throughput drop with no other symptom.
        env["cpu_affinity"] = len(os.sched_getaffinity(0))
    return env


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------


def make_optimizer(model: MiniGPT, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # The tied token embedding is a matrix and is decayed; the positional
        # table is excluded because every position is seen every step and
        # there is nothing to regularize toward.
        (decay if p.dim() >= 2 and name != "pos_emb.weight" else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=device.type == "cuda"
    )


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.lr * min_lr_ratio."""
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: MiniGPT, loader, device: torch.device, dtype: torch.dtype, micro_batch_size: int
) -> dict[str, float]:
    """Token-weighted mean cross-entropy and its perplexity over ``loader``."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for xm, ym in zip(x.split(micro_batch_size), y.split(micro_batch_size)):
            with autocast(device, dtype):
                _, loss = model(xm, ym)
            total_loss += loss.double() * ym.numel()
            total_tokens += ym.numel()
    model.train(was_training)
    mean = (total_loss / max(1, total_tokens)).item()
    return {"loss": mean, "ppl": math.exp(mean), "tokens": total_tokens}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_training_state(
    path: Path, model: MiniGPT, optimizer, scaler, cfg: TrainConfig, epoch: int, step: int,
    tokens_seen: int,
) -> None:
    """Everything needed to resume: weights, optimizer moments, loss scale and
    position in the run. Written to a temp file and renamed, so a crash
    mid-write cannot destroy the previous checkpoint."""
    state = {
        "format": "mini_gpt/training_state/v1",
        "model": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "train_config": asdict(cfg),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epochs_completed": epoch,
        "step": step,
        "tokens_seen": tokens_seen,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def export_checkpoint(path: Path, model: MiniGPT, cfg: TrainConfig, summary: dict) -> None:
    """A weights-only checkpoint for inference and grading: no optimizer
    state, so it is a third of the size of the training state."""
    torch.save(
        {
            "format": "mini_gpt/checkpoint/v1",
            "model": model.state_dict(),
            "model_config": model.cfg.to_dict(),
            "train_config": asdict(cfg),
            "summary": summary,
            "tokenizer": "gpt2",
        },
        path,
    )


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[MiniGPT, dict]:
    """Rebuild a MiniGPT from either checkpoint format. Returns (model, payload)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MiniGPT(GPTConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device), payload


# ---------------------------------------------------------------------------
# Metrics log
# ---------------------------------------------------------------------------


class MetricsLog:
    def __init__(self, path: Path, resume_after_step: int | None):
        self.path = path
        if resume_after_step is None:
            path.write_text("")
        else:
            # Drop records past the checkpoint: those steps are about to be
            # re-run, and keeping both copies would double them in the plots.
            kept = [
                line for line in path.read_text().splitlines()
                if line and json.loads(line)["step"] <= resume_after_step
            ]
            path.write_text("".join(line + "\n" for line in kept))
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, **record: Any) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build(cfg: TrainConfig):
    """Datasets, model and optimizer for ``cfg``, on the resolved device."""
    manifest = load_token_manifest(Path(cfg.tokens_dir))
    tokens_dir = Path(cfg.tokens_dir)
    train_ds = TokenWindowDataset(tokens_dir / "train.bin", cfg.seq_len, cfg.train_tokens, cfg.seed)
    # Evaluation windows use a fixed seed independent of the run seed, so
    # every run is scored on the same held-out tokens.
    val_ds = TokenWindowDataset(tokens_dir / "val.bin", cfg.seq_len, cfg.val_tokens, seed=0)
    full_val_ds = TokenWindowDataset(tokens_dir / "val.bin", cfg.seq_len, cfg.final_val_tokens or None)

    device = resolve_device(cfg.device)
    dtype = resolve_precision(cfg.precision, device)
    torch.manual_seed(cfg.seed)
    model = MiniGPT(cfg.model_config(manifest["tokenizer"]["vocab_size"])).to(device)
    optimizer = make_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    return manifest, train_ds, val_ds, full_val_ds, device, dtype, model, optimizer, scaler


def train(cfg: TrainConfig, resume: bool = False, export: Path | None = None) -> dict[str, Any]:
    if cfg.batch_size % cfg.micro_batch_size and cfg.micro_batch_size < cfg.batch_size:
        raise SystemExit("error: --batch-size must be a multiple of --micro-batch-size")
    micro = min(cfg.micro_batch_size, cfg.batch_size)

    manifest, train_ds, val_ds, full_val_ds, device, dtype, model, optimizer, scaler = build(cfg)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"

    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, drop_last=False, **loader_kw)
    steps_per_epoch = len(train_ds) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    tokens_per_step = cfg.batch_size * cfg.seq_len
    if steps_per_epoch == 0:
        raise SystemExit("error: train_tokens is smaller than one batch")

    start_epoch, step, tokens_seen = 0, 0, 0
    if resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        start_epoch, step, tokens_seen = state["epochs_completed"], state["step"], state["tokens_seen"]
        print(f"[{cfg.name}] resumed after epoch {start_epoch} (step {step})", flush=True)
    log = MetricsLog(run_dir / "metrics.jsonl", resume_after_step=step if start_epoch else None)

    params = model.num_parameters()
    (run_dir / "config.json").write_text(json.dumps({
        "train_config": asdict(cfg),
        "model_config": model.cfg.to_dict(),
        "parameters": params,
        "precision": str(dtype).removeprefix("torch."),
        "environment": environment(device),
        "data": {
            "tokens_dir": cfg.tokens_dir,
            "tokenizer": manifest["tokenizer"]["id"],
            "train_pages": len(train_ds.pages),
            "train_tokens_per_epoch": train_ds.n_tokens,
            "val_tokens_periodic": val_ds.n_tokens,
            "val_tokens_full": full_val_ds.n_tokens,
        },
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
    }, indent=2) + "\n")

    print(f"[{cfg.name}] {params['total'] / 1e6:.2f}M params on {device} "
          f"({str(dtype).removeprefix('torch.')}); {cfg.epochs} epochs x {steps_per_epoch:,} steps "
          f"of {cfg.batch_size}x{cfg.seq_len} tokens", flush=True)

    # Evaluate at evenly spaced token counts within each epoch.
    eval_steps = {
        e * steps_per_epoch + round(i * steps_per_epoch / cfg.evals_per_epoch)
        for e in range(cfg.epochs) for i in range(1, cfg.evals_per_epoch + 1)
    }

    if step == 0:
        # Step-0 evaluation anchors the curves at the untrained model, whose
        # loss should sit near ln(vocab_size) = 10.8.
        ev = evaluate(model, val_loader, device, dtype, micro)
        log.write(kind="eval", step=0, epoch=0, tokens=0, val_loss=ev["loss"], val_ppl=ev["ppl"])

    model.train()
    started = time.monotonic()
    for epoch in range(start_epoch, cfg.epochs):
        loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch, **loader_kw)
        epoch_loss_sum = torch.zeros((), device=device)
        window_loss = torch.zeros((), device=device)
        window_norm = torch.zeros((), device=device)
        window_norm_steps = torch.zeros((), device=device)
        window_steps, window_t0, window_eval_seconds = 0, time.monotonic(), 0.0
        epoch_t0 = time.monotonic()

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward and backward over micro-batches; each loss is scaled by
            # its share of the batch so the accumulated gradient equals the
            # gradient of the full-batch mean loss.
            step_loss = torch.zeros((), device=device)
            for xm, ym in zip(x.split(micro), y.split(micro)):
                with autocast(device, dtype):
                    _, loss = model(xm, ym)
                share = xm.size(0) / x.size(0)
                scaler.scale(loss * share).backward()
                step_loss += loss.detach() * share

            scaler.unscale_(optimizer)  # so clipping sees true gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            tokens_seen += tokens_per_step
            epoch_loss_sum += step_loss
            window_loss += step_loss
            # Under --precision fp16 the first steps can overflow; the scaler
            # skips them and the norm is inf, which would poison the window mean.
            finite = torch.isfinite(grad_norm)
            window_norm += torch.where(finite, grad_norm.float(), 0.0)
            window_norm_steps += finite.float()
            window_steps += 1

            # .item() forces a device sync, so it happens once per window,
            # not once per step.
            if step % cfg.log_every == 0 or step == total_steps:
                elapsed = time.monotonic() - window_t0 - window_eval_seconds
                mean_loss = (window_loss / window_steps).item()
                log.write(
                    kind="step", step=step, epoch=epoch + 1, tokens=tokens_seen, lr=lr,
                    loss=mean_loss, grad_norm=(window_norm / window_norm_steps.clamp(min=1)).item(),
                    tokens_per_sec=window_steps * tokens_per_step / max(elapsed, 1e-9),
                )
                window_loss.zero_()
                window_norm.zero_()
                window_norm_steps.zero_()
                window_steps, window_t0, window_eval_seconds = 0, time.monotonic(), 0.0

            if step in eval_steps:
                eval_t0 = time.monotonic()
                ev = evaluate(model, val_loader, device, dtype, micro)
                # Evaluation time is not training time; keep it out of tokens/s.
                window_eval_seconds += time.monotonic() - eval_t0
                log.write(kind="eval", step=step, epoch=epoch + 1, tokens=tokens_seen,
                          val_loss=ev["loss"], val_ppl=ev["ppl"])
                print(f"[{cfg.name}] epoch {epoch + 1} step {step:,}/{total_steps:,} "
                      f"tokens {tokens_seen / 1e6:,.1f}M  val loss {ev['loss']:.4f}  "
                      f"ppl {ev['ppl']:,.1f}  lr {lr:.2e}", flush=True)

        epoch_seconds = time.monotonic() - epoch_t0
        train_loss = (epoch_loss_sum / steps_per_epoch).item()
        ev = evaluate(model, val_loader, device, dtype, micro)
        log.write(
            kind="epoch", step=step, epoch=epoch + 1, tokens=tokens_seen,
            train_loss=train_loss, train_ppl=math.exp(train_loss),
            val_loss=ev["loss"], val_ppl=ev["ppl"], seconds=epoch_seconds,
        )
        print(f"[{cfg.name}] == epoch {epoch + 1}/{cfg.epochs}: avg train loss {train_loss:.4f} "
              f"(ppl {math.exp(train_loss):,.1f}), val loss {ev['loss']:.4f} (ppl {ev['ppl']:,.1f}), "
              f"{epoch_seconds:,.0f}s", flush=True)
        save_training_state(ckpt_path, model, optimizer, scaler, cfg, epoch + 1, step, tokens_seen)

    log.close()

    final = evaluate(model, make_loader(full_val_ds, cfg.batch_size, shuffle=False,
                                        drop_last=False, **loader_kw), device, dtype, micro)
    history = read_metrics(run_dir)
    epochs = [r for r in history if r["kind"] == "epoch"]
    rates = [r["tokens_per_sec"] for r in history if r["kind"] == "step"]
    summary = {
        "name": cfg.name,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "steps": step,
        "tokens_seen": tokens_seen,
        "epochs": [{k: r[k] for k in ("epoch", "train_loss", "train_ppl", "val_loss", "val_ppl")}
                   for r in epochs],
        "final_val_loss": final["loss"],
        "final_val_ppl": final["ppl"],
        "final_val_tokens": final["tokens"],
        "best_periodic_val_loss": min(r["val_loss"] for r in history if r["kind"] == "eval"),
        "median_tokens_per_sec": float(np.median(rates)) if rates else None,
        "train_seconds_this_session": round(time.monotonic() - started, 1),
        "environment": environment(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{cfg.name}] done: final val loss {final['loss']:.4f}, perplexity {final['ppl']:,.2f} "
          f"on {final['tokens']:,} tokens", flush=True)
    if export is not None:
        export_checkpoint(export, model, cfg, summary)
        print(f"[{cfg.name}] exported {export} ({export.stat().st_size / 1e6:,.1f} MB)", flush=True)
    return summary


def benchmark(cfg: TrainConfig, steps: int) -> dict[str, Any]:
    """Time ``steps`` optimizer steps after two warmup steps. No files written."""
    _, train_ds, _, _, device, dtype, model, optimizer, scaler = build(cfg)
    micro = min(cfg.micro_batch_size, cfg.batch_size)
    loader = iter(make_loader(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed,
                              num_workers=cfg.num_workers, pin_memory=device.type == "cuda"))
    model.train()

    def one_step() -> None:
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        for xm, ym in zip(x.split(micro), y.split(micro)):
            with autocast(device, dtype):
                _, loss = model(xm, ym)
            scaler.scale(loss * xm.size(0) / x.size(0)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(2):
        one_step()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    for _ in range(steps):
        one_step()
    sync()
    seconds = time.monotonic() - t0
    result = {
        "device": str(device),
        "precision": str(dtype).removeprefix("torch."),
        "parameters": model.num_parameters()["total"],
        "batch": f"{cfg.batch_size}x{cfg.seq_len} (micro {micro})",
        "seconds_per_step": seconds / steps,
        "tokens_per_sec": steps * cfg.batch_size * cfg.seq_len / seconds,
    }
    if device.type == "cuda":
        result["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        p.add_argument(flag, type=type(getattr(defaults, f.name)), default=getattr(defaults, f.name))
    p.add_argument("--resume", action="store_true", help="continue from <run>/checkpoint.pt")
    p.add_argument("--export", type=Path, help="also write a weights-only checkpoint here")
    p.add_argument("--benchmark-steps", type=int, default=0,
                   help="only measure throughput over this many steps, then exit")
    ns = p.parse_args(argv)
    cfg = TrainConfig(**{f.name: getattr(ns, f.name) for f in fields(TrainConfig)})
    return cfg, ns


def main(argv: list[str] | None = None) -> int:
    cfg, ns = parse_args(argv)
    if ns.benchmark_steps:
        print(json.dumps(benchmark(cfg, ns.benchmark_steps), indent=2))
        return 0
    train(cfg, resume=ns.resume, export=ns.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
