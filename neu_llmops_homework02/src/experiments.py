"""
Assignment 2: hyperparameter experiments.

A one-factor-at-a-time sweep around a baseline: each run changes exactly one
setting, so any difference from the baseline is attributable to that setting.
A full grid over the same five factors would take 162 runs; this takes 10.

Every run gets the **same token budget and the same training tokens**. That
is the fair comparison for batch size in particular: at a fixed budget a
larger batch means fewer, less noisy optimizer steps, which is the trade-off
worth measuring. Comparing at a fixed step count would hand the large-batch
run 4x the data.

Finished runs (``summary.json`` present) are skipped and interrupted ones are
resumed from their last epoch checkpoint, so re-running this script after a
crash only repeats the unfinished epoch.

After the sweep the run with the lowest final validation loss is exported as
the weights-only ``mini_gpt_checkpoint.pt``.

Usage
-----
    python src/experiments.py                        # the full sweep
    python src/experiments.py --only baseline lr_5e-4
    python src/experiments.py --smoke                # tiny budget, checks the plumbing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import TrainConfig, export_checkpoint, load_checkpoint, train  # noqa: E402

# (run name, factor it varies, overrides relative to the baseline)
SWEEP: list[tuple[str, str, dict]] = [
    ("baseline", "baseline", {}),
    ("lr_5e-4", "learning rate", {"lr": 5e-4}),
    ("lr_3e-3", "learning rate", {"lr": 3e-3}),
    ("batch_16", "batch size", {"batch_size": 16}),
    ("batch_64", "batch size", {"batch_size": 64}),
    ("layers_1", "layers", {"n_layer": 1}),
    ("embd_64", "embedding size", {"n_embd": 64}),
    ("embd_256", "embedding size", {"n_embd": 256}),
    ("seq_32", "sequence length", {"seq_len": 32}),
    ("seq_64", "sequence length", {"seq_len": 64}),
]

# Baseline value of each factor, for tables and plot legends.
FACTOR_KEYS = {
    "learning rate": "lr",
    "batch size": "batch_size",
    "layers": "n_layer",
    "embedding size": "n_embd",
    "sequence length": "seq_len",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--tokens-dir", help="Assignment 1 token directory (train.bin, val.bin, manifest.json)")
    p.add_argument("--train-tokens", type=int, help="override the baseline's per-epoch token budget")
    p.add_argument("--epochs", type=int, help="override the baseline's epoch count")
    p.add_argument("--micro-batch-size", type=int, help="largest batch run in one forward pass")
    p.add_argument("--device", default="auto")
    p.add_argument("--only", nargs="*", help="run only these names")
    p.add_argument("--checkpoint-out", type=Path, default=Path("mini_gpt_checkpoint.pt"))
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget into runs_smoke/, to validate the pipeline end to end")
    return p.parse_args(argv)


def baseline_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(out_dir=args.out_dir, device=args.device)
    if args.smoke:
        cfg = replace(cfg, out_dir="runs_smoke", train_tokens=64 * 1024, val_tokens=16 * 1024,
                      final_val_tokens=32 * 1024,
                      epochs=2, evals_per_epoch=2, log_every=2)
    overrides = {k: getattr(args, k) for k in ("tokens_dir", "train_tokens", "epochs", "micro_batch_size")
                 if getattr(args, k) is not None}
    return replace(cfg, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = baseline_config(args)
    names = {name for name, _, _ in SWEEP}
    unknown = set(args.only or []) - names
    if unknown:
        raise SystemExit(f"error: unknown run(s) {sorted(unknown)}; choose from {sorted(names)}")

    results = []
    for name, factor, overrides in SWEEP:
        if args.only and name not in args.only:
            continue
        cfg = replace(base, name=name, **overrides)
        summary_path = cfg.run_dir / "summary.json"
        if summary_path.is_file():
            print(f"[{name}] already complete; skipping", flush=True)
            summary = json.loads(summary_path.read_text())
        else:
            print(f"\n{'#' * 76}\n### {name}: {overrides or 'baseline settings'}\n{'#' * 76}", flush=True)
            summary = train(cfg, resume=True)
        results.append({"name": name, "factor": factor, "overrides": overrides,
                        "config": asdict(cfg), "summary": summary})

    out = Path(base.out_dir)
    sweep = {"baseline": asdict(base), "factor_keys": FACTOR_KEYS, "runs": results}
    (out / "sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")

    best = min(results, key=lambda r: r["summary"]["final_val_loss"])
    model, state = load_checkpoint(out / best["name"] / "checkpoint.pt")
    checkpoint_out = args.checkpoint_out if not args.smoke else out / args.checkpoint_out.name
    export_checkpoint(checkpoint_out, model, TrainConfig(**state["train_config"]), best["summary"])

    print(f"\n{'=' * 76}\n{'run':<12}{'factor':<18}{'val loss':>10}{'val ppl':>10}{'tok/s':>10}")
    for r in sorted(results, key=lambda r: r["summary"]["final_val_loss"]):
        s = r["summary"]
        print(f"{r['name']:<12}{r['factor']:<18}{s['final_val_loss']:>10.4f}"
              f"{s['final_val_ppl']:>10.1f}{s['median_tokens_per_sec'] or 0:>10,.0f}")
    print(f"\nbest: {best['name']} -> {checkpoint_out} "
          f"({checkpoint_out.stat().st_size / 1e6:,.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
