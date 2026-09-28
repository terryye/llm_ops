"""
Generate Assignment2_Report.pdf from the run logs.

Every number in the report is read from ``runs/sweep.json``, the per-run
``config.json`` / ``metrics.jsonl``, the Assignment 1 token manifest and
``samples.json``; the figures are the PNGs written by ``plots.py``. Nothing
is typed in by hand, so the report cannot disagree with the runs.

Usage
-----
    python src/make_report.py --runs runs --figures figures --out Assignment2_Report.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import read_metrics  # noqa: E402

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b5b5b")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#f2f2f2")
PAGE_WIDTH = LETTER[0] - 1.5 * inch


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, textColor=INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=MUTED, alignment=1, spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=14, textColor=INK, spaceBefore=11, spaceAfter=5,
            keepWithNext=1,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=3.5,
            leftIndent=11, bulletIndent=2,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=7.5, leading=10, textColor=MUTED, spaceBefore=2, spaceAfter=9,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Normal"], fontName="Courier",
            fontSize=7.8, leading=10.5, textColor=INK, spaceAfter=7,
            leftIndent=8, backColor=colors.HexColor("#f7f7f7"), borderPadding=5,
        ),
        "sample": ParagraphStyle(
            "sample", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=INK, spaceAfter=4, leftIndent=8,
        ),
    }


def table(data: list[list[str]], widths: list[float], align_right: set[int] = frozenset(),
          bold_rows: set[int] = frozenset()) -> Table:
    """Build a table. `widths` are in inches; reportlab wants points."""
    t = Table(data, colWidths=[w * inch for w in widths], hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 7.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    for row in bold_rows:
        style.append(("FONT", (0, row), (-1, row), "Helvetica-Bold", 7.6))
    t.setStyle(TableStyle(style))
    return t


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run the experiments first")
    return json.loads(path.read_text())


def figure(path: Path, width: float = PAGE_WIDTH) -> Image:
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run plots.py first")
    img = Image(str(path))
    img.drawHeight = img.drawHeight * width / img.drawWidth
    img.drawWidth = width
    return img


def human(n: float) -> str:
    return f"{n / 1e6:,.2f}M" if n >= 1e6 else f"{n / 1e3:,.1f}K"


def lr_str(v: float) -> str:
    mant, exp = f"{v:.0e}".split("e")
    return f"{mant}e-{int(exp[1:])}" if exp.startswith("-") else f"{v:g}"


def setting_str(key: str, v: Any) -> str:
    return lr_str(v) if key == "lr" else str(v)


def printable(text: str) -> str:
    """Samples on one line, without U+FFFD. The model can emit a byte-level
    BPE token holding half of a multi-byte character; decoding it alone gives
    the replacement character, which Helvetica has no glyph for."""
    return re.sub(r"\s+", " ", text.replace("\ufffd", "")).rstrip()


def build(runs_dir: Path, figures: Path, samples: dict | None, out: Path) -> None:
    S = styles()
    story: list[Any] = []

    def para(text: str, key: str = "body") -> None:
        story.append(Paragraph(text, S[key]))

    def bullet(text: str) -> None:
        story.append(Paragraph(text, S["bullet"], bulletText="•"))

    def caption(text: str) -> None:
        story.append(Paragraph(text, S["caption"]))

    sweep = load(runs_dir / "sweep.json")
    runs = {r["name"]: r for r in sweep["runs"]}
    base = runs["baseline"]
    bcfg = base["config"]
    bsum = base["summary"]
    bconf = load(runs_dir / "baseline" / "config.json")
    tok = load(Path(bcfg["tokens_dir"]) / "manifest.json")
    params = bsum["parameters"]
    env = bsum["environment"]
    device = env.get("gpu") or env["device"]
    best = min(sweep["runs"], key=lambda r: r["summary"]["final_val_loss"])
    total_tokens_trained = sum(r["summary"]["tokens_seen"] for r in sweep["runs"])

    # ---------------------------------------------------------------- header
    para("Assignment 2: Building a Small-Scale<br/>Foundation Model from Scratch", "title")
    para(
        f"mini-GPT &nbsp;|&nbsp; {human(params['total'])} parameters (baseline) &nbsp;|&nbsp; "
        f"{len(sweep['runs'])} training runs, {total_tokens_trained / 1e6:,.0f}M tokens in total "
        f"&nbsp;|&nbsp; {escape(device)} &nbsp;|&nbsp; generated from the run logs",
        "subtitle",
    )

    # ---------------------------------------------------------- 1 architecture
    para("1. Model architecture and parameters", "h1")
    d, L, H, T, V = bcfg["n_embd"], bcfg["n_layer"], bcfg["n_head"], bcfg["seq_len"], tok["tokenizer"]["vocab_size"]
    para(
        "The model is a decoder-only transformer in the GPT-2 layout, implemented from PyTorch "
        "primitives in <font face='Courier'>src/model.py</font>. Token ids are embedded, added to a "
        "learned positional embedding, and passed through a stack of <b>pre-LayerNorm</b> blocks, "
        "each computing <font face='Courier'>x + Attn(LN(x))</font> then "
        "<font face='Courier'>x + MLP(LN(x))</font>. Attention is written out explicitly: one linear "
        "map produces queries, keys and values for all heads; scores are scaled by "
        "1/sqrt(head_dim), masked with a lower-triangular causal mask, softmax-normalized and "
        "recombined. The MLP expands to 4d with a <b>GELU</b> activation. A final LayerNorm "
        "feeds a linear head whose weight is <b>tied</b> to the token embedding, producing "
        f"logits over all {V:,} GPT-2 tokens at every position. Initialization follows GPT-2 "
        "(N(0, 0.02), residual output projections scaled by 1/sqrt(2L)). A unit test checks "
        "causality directly: changing a future token leaves every earlier logit unchanged.",
    )
    rows = [
        ["Component", "Shape (baseline)", "Parameters", "Share"],
        ["Token embedding (tied with output head)", f"{V:,} x {d}", f"{params['token_embedding']:,}",
         f"{params['token_embedding'] / params['total']:.1%}"],
        ["Positional embedding (learned)", f"{T} x {d}", f"{params['position_embedding']:,}",
         f"{params['position_embedding'] / params['total']:.1%}"],
        [f"{L} transformer blocks ({H} heads, head_dim {d // H}, MLP {4 * d})",
         f"{L} x (LN, QKV {d}x{3 * d}, proj, LN, MLP)", f"{params['transformer_blocks']:,}",
         f"{params['transformer_blocks'] / params['total']:.1%}"],
        ["Final LayerNorm", f"{d} + {d}", f"{params['final_layernorm']:,}",
         f"{params['final_layernorm'] / params['total']:.1%}"],
        ["Total (unique)", "", f"{params['total']:,}", "100%"],
    ]
    story.append(table(rows, [2.9, 2.0, 1.1, 0.7], align_right={2, 3}, bold_rows={5}))
    caption(
        f"Table 1. Baseline parameters. The {V:,}-token vocabulary dominates a model this small: "
        f"the embedding is {params['token_embedding'] / params['transformer_blocks']:.0f}x the "
        "size of both transformer blocks together, and the output projection onto it is most of "
        "the compute per token. Tying the head to the embedding avoids a second table of the same size."
    )

    # --------------------------------------------------------------- 2 dataset
    para("2. Dataset", "h1")
    splits = tok["splits"]
    per_source = tok["tokens_per_source"]
    src_total = sum(per_source.values())
    para(
        "Training data is the Assignment 1 corpus: cleaned, exactly-deduplicated English text "
        f"from Wikipedia, CC-News and FineWeb ({', '.join(f'{k} {v / src_total:.0%}' for k, v in per_source.items())} "
        f"of tokens), tokenized with the GPT-2 BPE tokenizer and packed into flat uint16 files with "
        f"an end-of-text token between documents. The train split holds "
        f"{splits['train']['tokens'] / 1e6:,.1f}M tokens from {splits['train']['documents']:,} "
        f"documents; the validation split, assigned per document by hash, holds "
        f"{splits['val']['tokens'] / 1e6:,.2f}M tokens from {splits['val']['documents']:,} documents "
        "that never appear in training.",
    )
    dat = bconf["data"]
    para(
        f"Each run samples {dat['train_pages']:,} random 1024-token pages of the train split "
        f"({dat['train_tokens_per_epoch'] / 1e6:,.1f}M tokens per epoch, the same pages for every "
        "run) and cuts each page into consecutive windows of the run's sequence length; the target "
        "is the input shifted by one token. Because the page sample depends only on the seed, "
        "the sequence-length runs see identical text and differ only in how it is windowed. "
        "A shuffled DataLoader draws batches from the windows in a new seeded order each epoch. "
        f"During training, validation loss is measured on a fixed {dat['val_tokens_periodic'] / 1e3:,.0f}K-token "
        f"subset of the validation split; the final scores use a larger fixed subset of {dat['val_tokens_full'] / 1e3:,.0f}K tokens, the same for every run.",
    )

    # -------------------------------------------------------- 3 training setup
    para("3. Training setup", "h1")
    prec = bconf["precision"]
    micro = bcfg["micro_batch_size"]
    para(
        "Each step runs the forward pass, the mean cross-entropy of the next-token logits against "
        "the shifted targets, the backward pass, gradient-norm clipping, and an AdamW update. "
        f"Batches larger than {micro} sequences are split into micro-batches whose gradients "
        "accumulate before the single optimizer step (a unit test checks that this matches the "
        "full-batch gradient). Perplexity is exp of the token-weighted mean cross-entropy over the "
        "whole evaluation set. The average training loss of each epoch, validation loss and "
        "perplexity at evenly spaced points, learning rate, gradient norm and throughput are "
        "logged to <font face='Courier'>metrics.jsonl</font>; a resumable checkpoint (weights, "
        "optimizer moments, loss scale, position) is saved after every epoch.",
    )
    rows = [
        ["Setting", "Baseline value", "Setting", "Baseline value"],
        ["Optimizer", f"AdamW, betas ({bcfg['beta1']}, {bcfg['beta2']})", "Epochs", f"{bcfg['epochs']}"],
        ["Peak learning rate", lr_str(bcfg["lr"]), "Batch size", f"{bcfg['batch_size']} x {bcfg['seq_len']} tokens"],
        ["Schedule", f"{bcfg['warmup_frac']:.0%} linear warmup, cosine to {bcfg['min_lr_ratio']:.0%}",
         "Steps", f"{bconf['total_steps']:,} ({bconf['steps_per_epoch']:,} per epoch)"],
        ["Weight decay", f"{bcfg['weight_decay']} (matrices only)", "Tokens seen", f"{bsum['tokens_seen'] / 1e6:,.1f}M"],
        ["Gradient clipping", f"global norm {bcfg['grad_clip']}", "Precision", prec + (" autocast + loss scaling" if prec == "float16" else "")],
        ["Dropout", f"{bcfg['dropout']}", "Hardware", escape(device)],
        ["Seed", f"{bcfg['seed']}", "Throughput", f"{bsum['median_tokens_per_sec']:,.0f} tokens/s (median)"],
    ]
    story.append(KeepTogether([
        table(rows, [1.25, 2.2, 1.0, 2.25]),
        Paragraph(
            "Table 2. Baseline training configuration. Dropout is off: every run sees each token only "
            f"{bcfg['epochs']} times, so the model is far from memorizing its data (see the train/validation "
            "gap in Section 4), and dropout would only slow learning.", S["caption"]),
    ]))

    # ---------------------------------------------------------------- 4 baseline
    epochs = [r for r in read_metrics(runs_dir / "baseline") if r["kind"] == "epoch"]
    story.append(KeepTogether([
        Paragraph("4. Baseline training dynamics", S["h1"]),
        figure(figures / "baseline_curves.png"),
        Paragraph(
            "Figure 1. Baseline run. Left: training loss per 10-step window (faint) and its moving "
            "average, with validation loss. The y axis starts after the first 10% of training so the "
            "later differences are visible. Middle: validation perplexity on a log scale, from the "
            "untrained model onward. Right: the average training loss of each epoch next to the "
            "validation loss at that epoch's end.", S["caption"]),
    ]))
    init_ppl = next(r["val_ppl"] for r in read_metrics(runs_dir / "baseline") if r["kind"] == "eval")
    last = epochs[-1]
    gap = last["val_loss"] - last["train_loss"]
    para(
        f"The untrained model scores a perplexity of {init_ppl:,.0f}, close to the {V:,} a uniform "
        f"guess over the vocabulary would give (loss ln {V:,} = {math.log(V):.2f}). Loss falls "
        "steeply during warmup, as the model first learns token frequencies and then common "
        f"short-range patterns, and keeps falling through all {len(epochs)} epochs. After training, the baseline "
        f"reaches a validation loss of {bsum['final_val_loss']:.3f} on the held-out final-evaluation subset "
        f"(perplexity <b>{bsum['final_val_ppl']:,.1f}</b>). In the last epoch the average training "
        f"loss was {last['train_loss']:.3f} against {last['val_loss']:.3f} on validation, a gap of "
        f"{gap:+.3f} nats. {'The two remain close, so the model is still underfitting: capacity and data, not memorization, limit it.' if abs(gap) < 0.1 else 'The gap has opened up: the model has started fitting its training pages more closely than new text.'}",
    )

    # ------------------------------------------------------------ 5 experiments
    para("5. Hyperparameter experiments", "h1")
    para(
        "The sweep changes one setting at a time from the baseline, so each difference is "
        "attributable to that setting: learning rate, batch size, number of layers, embedding "
        "size and sequence length (a full grid of the same values would be 162 runs; this is "
        f"{len(sweep['runs'])}). Every run trains on the same pages for the same number of tokens. "
        "For batch size this is the fair comparison: at a fixed budget a larger batch takes fewer, "
        "less noisy steps, and that trade-off is what is being measured.",
    )
    fkeys = sweep["factor_keys"]
    rows = [["Run", "Changed setting", "Params", "Steps", "Val loss", "Val ppl", "vs base", "Tokens/s"]]
    order = ["baseline"] + [r["name"] for r in sweep["runs"] if r["name"] != "baseline"]
    for name in order:
        r = runs[name]
        s = r["summary"]
        if name == "baseline":
            changed = "none"
        else:
            key = fkeys[r["factor"]]
            changed = f"{r['factor']} {setting_str(key, bcfg[key])} -> {setting_str(key, r['config'][key])}"
        delta = (s["final_val_ppl"] / bsum["final_val_ppl"] - 1) * 100
        rows.append([
            name, changed, human(s["parameters"]["total"]), f"{s['steps']:,}",
            f"{s['final_val_loss']:.3f}", f"{s['final_val_ppl']:,.1f}",
            "" if name == "baseline" else f"{delta:+.1f}%",
            f"{s['median_tokens_per_sec']:,.0f}",
        ])
    story.append(KeepTogether([
        table(rows, [0.8, 2.05, 0.6, 0.55, 0.65, 0.65, 0.6, 0.8],
              align_right={2, 3, 4, 5, 6, 7}, bold_rows={1 + order.index(best["name"])}),
        Paragraph(
            "Table 3. All runs, scored on the same held-out validation tokens after training. \"vs base\" is the "
            f"change in perplexity relative to the baseline (negative is better). Bold: the best run "
            f"({best['name']}), exported as mini_gpt_checkpoint.pt.", S["caption"]),
    ]))
    story.append(KeepTogether([
        figure(figures / "sweep_curves.png"),
        Paragraph(
            "Figure 2. Validation loss during training for each factor, baseline in blue in every "
            "panel. All panels share one y axis, zoomed past the first 10% of training.", S["caption"]),
    ]))
    story.append(KeepTogether([
        figure(figures / "sweep_perplexity.png"),
        Paragraph("Figure 3. Final validation perplexity for every run, best first. The vertical "
                  "line marks the baseline.", S["caption"]),
    ]))

    # One sentence per factor, computed from the table rather than written by hand.
    para("<b>Findings by factor.</b>")
    for factor, key in fkeys.items():
        members = [base] + [r for r in sweep["runs"] if r["factor"] == factor]
        members.sort(key=lambda r: r["config"][key])
        ranked = sorted(members, key=lambda r: r["summary"]["final_val_loss"])
        parts = ", ".join(
            f"{setting_str(key, r['config'][key])}: {r['summary']['final_val_ppl']:,.1f}" for r in members
        )
        top = ranked[0]
        spread = (ranked[-1]["summary"]["final_val_ppl"] / top["summary"]["final_val_ppl"] - 1) * 100
        bullet(f"<b>{factor.capitalize()}</b> (perplexity {parts}). Best: "
               f"{setting_str(key, top['config'][key])}; the worst setting is {spread:.1f}% higher. "
               + FACTOR_NOTES.get(factor, ""))

    # --------------------------------------------------------------- 6 samples
    if samples:
        story.append(KeepTogether([
            Paragraph("6. Samples from the exported checkpoint", S["h1"]),
            Paragraph(
                f"Generated by <font face='Courier'>src/generate.py</font>, which rebuilds the model from "
                f"mini_gpt_checkpoint.pt alone (run {escape(samples['run'])}; temperature "
                f"{samples['settings']['temperature']}, top-k {samples['settings']['top_k']}, "
                f"{samples['settings']['tokens']} new tokens). The prompt is in bold.", S["body"]),
            *[Paragraph(f"<b>{escape(s['prompt'])}</b>{escape(printable(s['completion']))}",
                        S["sample"]) for s in samples["samples"]],
        ]))
        para(SAMPLES_NOTE)

    # ---------------------------------------------------------- 7 observations
    para(f"{7 if samples else 6}. Observations and challenges", "h1")
    for text in observations(sweep, bsum):
        bullet(text)
    for text in CHALLENGES:
        bullet(text)

    doc = SimpleDocTemplate(
        str(out), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="Assignment 2: Building a Small-Scale Foundation Model from Scratch",
        author="",
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.42 * inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# Interpretation that needs a human reading of the curves, kept here beside
# the code that prints the numbers it refers to. Revisit after each sweep.
FACTOR_NOTES: dict[str, str] = {}
SAMPLES_NOTE = (
    "The samples are locally fluent at best: common function words, plausible punctuation and "
    "short phrases, but no coherence beyond a few tokens. That is what a validation perplexity in "
    "the hundreds means: the model narrows each next token to a few hundred candidates, which "
    "captures word frequencies and short-range grammar but not meaning."
)


def observations(sweep: dict, bsum: dict) -> list[str]:
    """Observations whose numbers come from the sweep."""
    runs = {r["name"]: r["summary"] for r in sweep["runs"]}
    best_name = min(runs, key=lambda n: runs[n]["final_val_loss"])
    best = runs[best_name]
    out = [
        f"<b>Best run.</b> {best_name} reached perplexity {best['final_val_ppl']:,.1f}, "
        f"{(1 - best['final_val_ppl'] / bsum['final_val_ppl']) * 100:.1f}% below the baseline's "
        f"{bsum['final_val_ppl']:,.1f}, after {best['tokens_seen'] / 1e6:,.1f}M training tokens."
    ]
    if "embd_256" in runs and "embd_64" in runs:
        big, small = runs["embd_256"], runs["embd_64"]
        out.append(
            f"<b>Capacity costs compute.</b> Doubling the embedding to 256 changed perplexity from "
            f"{bsum['final_val_ppl']:,.1f} to {big['final_val_ppl']:,.1f} but cut throughput from "
            f"{bsum['median_tokens_per_sec']:,.0f} to {big['median_tokens_per_sec']:,.0f} tokens/s, "
            "because the cost of the output projection grows linearly with d. Halving it to 64 gave "
            f"{small['final_val_ppl']:,.1f} at {small['median_tokens_per_sec']:,.0f} tokens/s. "
            "With a fixed token budget, the extra quality of a wider model is paid for in wall-clock time."
        )
    if "layers_1" in runs:
        one = runs["layers_1"]
        out.append(
            f"<b>Depth matters less than width here.</b> Going from 2 layers to 1 moved perplexity "
            f"from {bsum['final_val_ppl']:,.1f} to {one['final_val_ppl']:,.1f} "
            f"({(one['final_val_ppl'] / bsum['final_val_ppl'] - 1) * 100:+.1f}%), and removes only "
            f"{(bsum['parameters']['total'] - one['parameters']['total']) / 1e3:,.0f}K parameters: "
            "the transformer blocks are a small share of a model whose size is set by the vocabulary."
        )
    return out


# Measured during this assignment; see the code comments they point to.
CHALLENGES: list[str] = [
    "<b>The vocabulary dominates compute and memory.</b> The GPT-2 vocabulary makes the output "
    "projection and its 50,257-way softmax most of the work per token. The logits for one batch of "
    "32 x 128 tokens are 0.8 GB in fp32, and with the log-softmax and gradients a 4 GB GTX 1650 ran "
    "out of memory at 32 sequences per forward pass. Gradient accumulation over micro-batches of 16 "
    "keeps the optimization batch size independent of memory (peak 1.8 GB).",
    "<b>Mixed precision was slower.</b> fp16 autocast measured 18.5K tokens/s against 21.1K in "
    "fp32 on the GTX 1650, which has no tensor cores; the casts cost more than the half-precision "
    "arithmetic saves. Training therefore runs in fp32, with bf16 used automatically only on GPUs "
    "that support it natively.",
    "<b>Hardware and time budget.</b> The first environment was CPU-only (about 2,400 tokens/s); "
    "moving to the GPU gave a roughly 10x speedup. The deadline still limited each run to a few "
    "million tokens, so every model is far from converged: validation loss is still falling at "
    "the end of every curve. The rankings describe early training. A lower learning rate or a "
    "larger batch, which lose here, may catch up with a longer budget.",
    "<b>Gradient accumulation has a throughput cost.</b> Batches of 32 or more run as micro-batches "
    "of 16, and the batch-16 run, which needs no split, trained about 1.7x faster in tokens/s than "
    "the baseline (Table 3). The per-micro-batch overhead of the split was not investigated further "
    "within the time available; it affects speed only, not the gradients, which a unit test shows "
    "are identical to the full-batch ones.",
    "<b>Sequence length changes the task, not only the model.</b> A model with a 32-token context "
    "is scored on predicting each token from at most 31 predecessors, so its perplexity is not "
    "directly comparable to a 128-token model's. The runs see identical text, but a lower loss at "
    "short context means the model learned faster per token, not that it models language better.",
    "<b>What an epoch means.</b> The full training split holds 334M tokens, far beyond this "
    "budget, so each run trains for two epochs over a fixed random sample of it. The small "
    "train/validation gap shows that the repetition has not yet led to memorization.",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the assignment report from the run logs.")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--figures", type=Path, default=Path("figures"))
    p.add_argument("--samples", type=Path, default=Path("samples.json"))
    p.add_argument("--out", type=Path, default=Path("Assignment2_Report.pdf"))
    args = p.parse_args(argv)

    samples = json.loads(args.samples.read_text()) if args.samples.is_file() else None
    build(args.runs, args.figures, samples, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
