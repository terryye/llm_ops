"""
Assignment 2: training-dynamics figures, drawn from the run logs.

    figures/baseline_curves.png   baseline loss, perplexity and per-epoch averages
    figures/sweep_curves.png      validation loss per hyperparameter, one panel each
    figures/sweep_perplexity.png  final validation perplexity of every run

Every number comes from ``runs/*/metrics.jsonl`` and ``runs/sweep.json``, so
the figures cannot drift from the runs they describe.

Style: thin 2px lines, hairline solid gridlines, text in ink colors rather
than series colors, and a fixed categorical order (the baseline is always
slot 1, so it keeps its color across panels). The three categorical slots
used here pass colorblind-separation checks for all pairs; slot 3 is under
3:1 contrast on the light surface, so every panel also carries direct labels.

Usage
-----
    python src/plots.py --runs runs --out figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import read_metrics  # noqa: E402

SURFACE = "#ffffff"  # the report page; figures sit on it without a visible box
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua; fixed order

LINE = 1.5  # points; ~2px at the report's print size
MARKER = 5.5


def sans_font() -> str:
    """The first installed of the report's sans faces. Passing matplotlib the
    whole list instead logs a warning for every missing family on every text
    element."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Helvetica", "Arial", "Liberation Sans") if f in installed), "DejaVu Sans")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": sans_font(),
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 7.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.labelcolor": INK_2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def millions(x: float, _pos: Any = None) -> str:
    if x == 0:
        return "0"
    return f"{x / 1e6:,.0f}M" if x >= 1e6 else f"{x / 1e3:,.0f}K"


def commas(x: float, _pos: Any = None) -> str:
    return f"{x:,.0f}"


def ema(values: list[float], alpha: float = 0.1) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def split(records: list[dict]) -> dict[str, list[dict]]:
    return {k: [r for r in records if r["kind"] == k] for k in ("step", "eval", "epoch")}


def end_label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(5, dy), textcoords="offset points",
                va="center", ha="left", fontsize=7, color=INK_2)


def marker_kw(color: str) -> dict:
    # A surface-colored ring keeps markers legible where they sit on a line.
    return dict(marker="o", markersize=MARKER, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2)


def zoom_ylim(ax, series: list[list[float]], pad: float = 0.06) -> None:
    lo = min(min(s) for s in series)
    hi = max(max(s) for s in series)
    span = hi - lo or 1.0
    ax.set_ylim(lo - pad * span, hi + pad * span)


# ---------------------------------------------------------------------------
# Figure 1: baseline
# ---------------------------------------------------------------------------


def baseline_figure(run_dir: Path, out: Path) -> None:
    m = split(read_metrics(run_dir))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})

    # (a) loss vs tokens: training (per log window, smoothed) and validation.
    ax = axes[0]
    tx = [r["tokens"] for r in m["step"]]
    ty = [r["loss"] for r in m["step"]]
    ax.plot(tx, ty, color=SERIES[0], lw=0.6, alpha=0.25)
    ax.plot(tx, ema(ty), color=SERIES[0], lw=LINE)
    ev = [r for r in m["eval"] if r["step"] > 0]
    vx, vy = [r["tokens"] for r in ev], [r["val_loss"] for r in ev]
    ax.plot(vx, vy, color=SERIES[1], lw=LINE, **marker_kw(SERIES[1]))
    # Zoom past the first ~10% of training, where loss falls from ~10.8 and
    # would otherwise flatten everything after it.
    cut = 0.1 * tx[-1]
    zoom_ylim(ax, [[y for x, y in zip(tx, ty) if x > cut], [y for x, y in zip(vx, vy) if x > cut]])
    ax.set_title("Cross-entropy loss")
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Loss (nats / token)")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    end_label(ax, vx[-1], vy[-1], f"{vy[-1]:.2f}")

    # (b) validation perplexity; log scale because it starts near 50,000.
    ax = axes[1]
    ev0 = m["eval"]
    px, py = [r["tokens"] for r in ev0], [r["val_ppl"] for r in ev0]
    ax.plot(px, py, color=SERIES[1], lw=LINE, **marker_kw(SERIES[1]))
    ax.set_yscale("log")
    ax.set_title("Validation perplexity")
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Perplexity (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    # 1-2-5 ticks: a plain log axis labels only the powers of ten, which
    # over this range leaves a single labelled tick.
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(commas))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.annotate(f"{py[0]:,.0f} at init", (px[0], py[0]), xytext=(6, -2), textcoords="offset points",
                fontsize=7, color=INK_2, va="top")
    end_label(ax, px[-1], py[-1], f"{py[-1]:,.0f}", dy=6)

    # (c) per-epoch averages: train loss is the mean over the epoch's steps,
    # validation is measured at the epoch's end.
    ax = axes[2]
    epochs = m["epoch"]
    xs = list(range(len(epochs)))
    w = 0.34
    for i, key in enumerate(("train_loss", "val_loss")):
        vals = [r[key] for r in epochs]
        pos = [x + (i - 0.5) * (w + 0.03) for x in xs]
        ax.bar(pos, vals, width=w, color=SERIES[i], zorder=2)
        for p, v in zip(pos, vals):
            ax.text(p, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=INK_2)
    lo = min(min(r["train_loss"], r["val_loss"]) for r in epochs)
    hi = max(max(r["train_loss"], r["val_loss"]) for r in epochs)
    # Truncated baseline, stated in the y label: the differences between
    # epochs are a few percent of the loss and vanish on a zero-based axis.
    ax.set_ylim(max(0, lo - 0.6 * (hi - lo) - 0.3), hi + 0.12 * (hi - lo) + 0.1)
    ax.set_xticks(xs, [f"Epoch {r['epoch']}" for r in epochs])
    ax.set_title("Loss per epoch")
    ax.set_ylabel("Loss (nats / token, axis cut)")
    ax.grid(axis="x", visible=False)
    # One legend for all three panels: blue is training and orange is
    # validation throughout. Below the plots, where it covers no data.
    handles = [
        Line2D([], [], color=SERIES[0], lw=LINE, label="Training loss (smoothed; epoch mean in the bars)"),
        Line2D([], [], color=SERIES[1], lw=LINE, label="Validation", **marker_kw(SERIES[1])),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.6, rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: sweep, one panel per factor
# ---------------------------------------------------------------------------


def fmt_value(key: str, value: Any) -> str:
    if key == "lr":
        mant, exp = f"{value:.0e}".split("e")
        return f"lr {mant}e-{int(exp[1:])}" if exp.startswith("-") else f"lr {value:g}"
    return {"batch_size": "batch {}", "n_layer": "{} layer", "n_embd": "d = {}",
            "seq_len": "T = {}"}[key].format(value) + ("s" if key == "n_layer" and value != 1 else "")


def sweep_figure(runs_dir: Path, sweep: dict, out: Path) -> None:
    base_run = next(r for r in sweep["runs"] if r["name"] == "baseline")
    factors = list(sweep["factor_keys"].items())
    fig, axes = plt.subplots(1, len(factors), figsize=(7.4, 2.6), sharey=True)
    all_y: list[float] = []
    panels = []
    for ax, (factor, key) in zip(axes, factors):
        variants = sorted((r for r in sweep["runs"] if r["factor"] == factor),
                          key=lambda r: r["config"][key])
        members = [base_run] + variants  # baseline first: it always takes slot 1
        panels.append((ax, factor, key, members))
        for r in members:
            ev = [e for e in read_metrics(runs_dir / r["name"]) if e["kind"] == "eval" and e["step"] > 0]
            all_y += [e["val_loss"] for e in ev if e["tokens"] > 0.1 * ev[-1]["tokens"]]

    for ax, factor, key, members in panels:
        for slot, r in enumerate(members):
            ev = [e for e in read_metrics(runs_dir / r["name"]) if e["kind"] == "eval" and e["step"] > 0]
            xs, ys = [e["tokens"] for e in ev], [e["val_loss"] for e in ev]
            label = fmt_value(key, r["config"][key]) + (" (baseline)" if r["name"] == "baseline" else "")
            ax.plot(xs, ys, color=SERIES[slot], lw=LINE, label=label)
            ax.plot(xs[-1:], ys[-1:], color=SERIES[slot], lw=0, **marker_kw(SERIES[slot]))
        ax.set_title(factor.capitalize())
        ax.set_xlabel("Tokens seen")
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        # Legends go under each panel: inside, they would sit on the curves.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.2)
    zoom_ylim(axes[0], [all_y], pad=0.07)
    axes[0].set_ylabel("Validation loss")

    fig.tight_layout(w_pad=0.8)
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: final perplexity per run
# ---------------------------------------------------------------------------


def perplexity_figure(sweep: dict, out: Path) -> None:
    runs = sorted(sweep["runs"], key=lambda r: r["summary"]["final_val_ppl"])
    names = [r["name"] for r in runs]
    ppl = [r["summary"]["final_val_ppl"] for r in runs]
    base = next(r["summary"]["final_val_ppl"] for r in runs if r["name"] == "baseline")

    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(runs) + 0.75))
    ys = list(range(len(runs)))[::-1]
    # One series, one color. The baseline is marked by a reference line and a
    # bold label rather than a second hue.
    ax.barh(ys, ppl, height=0.62, color=SERIES[0], zorder=2)
    ax.axvline(base, color=INK_2, lw=0.8, zorder=1)
    ax.annotate("baseline", (base, ys[0] + 0.62), xytext=(3, 0), textcoords="offset points",
                fontsize=6.5, color=INK_2, va="center")
    for y, v in zip(ys, ppl):
        delta = (v / base - 1) * 100
        tag = "" if math.isclose(v, base) else f"  ({delta:+.1f}%)"
        # A surface-colored box so the baseline rule never runs through a label.
        ax.text(v, y, f" {v:,.1f}{tag}", va="center", ha="left", fontsize=6.8, color=INK_2, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none"))
    ax.set_yticks(ys, names)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "baseline":
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.set_xlim(0, max(ppl) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(commas))
    ax.set_xlabel("Final validation perplexity (lower is better)")
    ax.set_title("Final validation perplexity by run")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Draw training-dynamics figures from run logs.")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args(argv)

    sweep_path = args.runs / "sweep.json"
    if not sweep_path.is_file():
        raise SystemExit(f"error: {sweep_path} not found; run experiments.py first")
    sweep = json.loads(sweep_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    baseline_figure(args.runs / "baseline", args.out / "baseline_curves.png")
    sweep_figure(args.runs, sweep, args.out / "sweep_curves.png")
    perplexity_figure(sweep, args.out / "sweep_perplexity.png")
    for f in ("baseline_curves.png", "sweep_curves.png", "sweep_perplexity.png"):
        print(f"wrote {args.out / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
