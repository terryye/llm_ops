#!/usr/bin/env python3
"""
Assignment 2: build a small-scale foundation model from scratch.

Single entry point for the deliverables. Runs, in order:

    1. src/experiments.py   baseline + one-factor hyperparameter sweep  -> runs/,
                            best run exported as mini_gpt_checkpoint.pt
    2. src/plots.py         loss and perplexity curves                 -> figures/
    3. src/generate.py      reload the checkpoint and sample text      -> samples.json
    4. src/make_report.py   report from the logs and figures           -> Assignment2_Report.pdf

Every stage also runs standalone. Finished runs are skipped and interrupted
ones resume from their last epoch, so re-running after a crash is cheap.

Usage
-----
    python train_mini_gpt.py                  # full run (use a GPU)
    python train_mini_gpt.py --smoke          # tiny budget into runs_smoke/, ~10 min on CPU
    python train_mini_gpt.py --skip train     # redraw figures and report only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STAGES = ("train", "plots", "samples", "report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    p.add_argument("--smoke", action="store_true", help="tiny budget; outputs go to *_smoke paths")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def banner(n: int, name: str) -> float:
    print(f"\n{'#' * 76}\n### Stage {n}: {name}\n{'#' * 76}", flush=True)
    return time.monotonic()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import experiments
    import generate
    import make_report
    import plots

    suffix = "_smoke" if args.smoke else ""
    runs = Path(f"runs{suffix}")
    figures = Path(f"figures{suffix}")
    checkpoint = runs / "mini_gpt_checkpoint.pt" if args.smoke else Path("mini_gpt_checkpoint.pt")
    samples = Path(f"samples{suffix}.json")
    report = Path(f"Assignment2_Report{suffix}.pdf")
    timings: list[tuple[str, float]] = []

    stages = [
        ("train", "hyperparameter sweep -> " + str(runs),
         lambda: experiments.main(["--device", args.device] + (["--smoke"] if args.smoke else []))),
        ("plots", f"figures -> {figures}",
         lambda: plots.main(["--runs", str(runs), "--out", str(figures)])),
        ("samples", f"reload {checkpoint} and sample -> {samples}",
         lambda: generate.main(["--checkpoint", str(checkpoint), "--device", args.device,
                                "--out", str(samples)])),
        ("report", f"report -> {report}",
         lambda: make_report.main(["--runs", str(runs), "--figures", str(figures),
                                   "--samples", str(samples), "--out", str(report)])),
    ]
    for n, (name, title, fn) in enumerate(stages, 1):
        if name in args.skip:
            continue
        t = banner(n, title)
        rc = fn()
        if rc != 0:
            return rc
        timings.append((name, time.monotonic() - t))

    print(f"\n{'=' * 76}\nDone.")
    for name, secs in timings:
        print(f"  {name:<8} {secs / 60:6.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
