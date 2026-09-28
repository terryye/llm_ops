"""
Assignment 2: a mini-GPT, a decoder-only transformer for next-token prediction.

Architecture (GPT-2 style, pre-LayerNorm):

    token ids (B, T)
      -> token embedding (V, d) + learned positional embedding (T_max, d)
      -> n_layer x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]
      -> final LayerNorm
      -> linear head tied to the token embedding -> logits (B, T, V)

Design notes
------------
* **Attention is written out by hand** rather than calling
  ``nn.MultiheadAttention`` or ``F.scaled_dot_product_attention``, so every
  step the assignment grades (projections, head split, scaled scores, causal
  mask, softmax, recombination) is visible. At T <= 128 the attention matrix is
  tiny next to the vocabulary projection, so a fused kernel would save nothing
  measurable.
* **Pre-LN, not post-LN.** Normalizing the input of each sublayer keeps the
  residual stream an identity path, which trains stably without the careful
  warmup that post-LN (the original Transformer) needs.
* **GELU** in the MLP, as in GPT-2; its smooth gate trains slightly better
  than ReLU for language models and costs the same.
* **Weight tying.** GPT-2's 50,257-token vocabulary makes the embedding table
  the largest tensor in a model this small (6.4M parameters at d=128, against
  0.4M for two transformer blocks). Sharing it with the output head halves
  that cost and ties the input and output meaning of each token together,
  which helps most when, as here, the model sees each rare token only a few
  times.
* **Initialization** follows GPT-2: N(0, 0.02) everywhere, with the two
  projections that write into the residual stream scaled down by
  1/sqrt(2 * n_layer) so the stream's variance does not grow with depth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer, as used in Assignment 1
    block_size: int = 128  # maximum context length; sizes the positional table
    n_layer: int = 2
    n_embd: int = 128
    n_head: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd={self.n_embd} is not divisible by n_head={self.n_head}")

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t attends only to positions <= t."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One matmul produces queries, keys and values for all heads at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask; a buffer so it follows .to(device) but is not
        # a parameter and is not saved in checkpoints.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores. Dividing by sqrt(head_dim) keeps their
        # variance near 1, so the softmax does not saturate at initialization.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))

        out = weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network with the conventional 4x expansion."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One pre-LN transformer block: attention then MLP, each on a residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying; see module docstring

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> dict[str, int]:
        """Parameter counts by component. The tied head is counted once."""
        emb = self.tok_emb.weight.numel()
        pos = self.pos_emb.weight.numel()
        blocks = sum(p.numel() for p in self.blocks.parameters())
        final_ln = sum(p.numel() for p in self.ln_f.parameters())
        return {
            "token_embedding": emb,
            "position_embedding": pos,
            "transformer_blocks": blocks,
            "final_layernorm": final_ln,
            "total": emb + pos + blocks + final_ln,
        }

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits (B, T, V) and, if targets are given, the
        mean cross-entropy over all B*T positions."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            # Upcast so the log-softmax over 50k classes is computed in fp32
            # under reduced-precision autocast (a no-op, not a copy, in fp32).
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens after ``idx`` (B, T)."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
