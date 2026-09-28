"""
Assignment 2: load a checkpoint and sample text from it.

Demonstrates the load side of checkpointing: the model is rebuilt from the
config stored inside the file, so no training code or command-line flags are
needed to use it. Samples are written to JSON for the report.

Usage
-----
    python src/generate.py --checkpoint mini_gpt_checkpoint.pt
    python src/generate.py --checkpoint mini_gpt_checkpoint.pt --prompt "The city of" --tokens 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_checkpoint, resolve_device  # noqa: E402

DEFAULT_PROMPTS = [
    "The history of the city",
    "Scientists announced on Tuesday that",
    "In computer science, a",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sample text from a mini-GPT checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=Path("mini_gpt_checkpoint.pt"))
    p.add_argument("--prompt", action="append", help="repeatable; defaults to three fixed prompts")
    p.add_argument("--tokens", type=int, default=40, help="new tokens per sample")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, help="write samples as JSON here")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(payload.get("tokenizer", "gpt2"))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    samples = []
    for prompt in args.prompt or DEFAULT_PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(ids, args.tokens, temperature=args.temperature,
                             top_k=args.top_k, generator=generator)
        text = tokenizer.decode(out[0, ids.size(1):].tolist())
        samples.append({"prompt": prompt, "completion": text})
        print(f"--- {prompt!r}\n{prompt}{text}\n")

    if args.out:
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "run": payload["train_config"]["name"],
            "settings": {"tokens": args.tokens, "temperature": args.temperature,
                         "top_k": args.top_k, "seed": args.seed},
            "samples": samples,
        }, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
