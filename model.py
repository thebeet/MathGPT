from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, n_head, T, head_dim)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Cached decode: q is one token attending to full history — no causal mask needed.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, d_ff: int, dropout: float, max_seq_len: int
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class MathGPT(nn.Module):
    """Minimal decoder-only Transformer (GPT-style) for equation next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout, max_seq_len) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, idx: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(past_len, past_len + t, device=idx.device)
        return self.drop(self.tok_emb(idx) + self.pos_emb(pos))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self._embed(idx)
        for block in self.blocks:
            x, _ = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def _forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if past_len + idx.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Cached sequence length {past_len + idx.shape[1]} "
                f"exceeds max_seq_len={self.max_seq_len}"
            )

        x = self._embed(idx, past_len=past_len)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = block(x, past_kv)
            new_cache.append(new_kv)
        logits = self.lm_head(self.ln_f(x))
        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive generation with KV cache."""
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        for _ in range(max_new_tokens):
            if idx.shape[1] >= self.max_seq_len:
                break
            if past is None:
                idx_cond = idx[:, -self.max_seq_len :]
            else:
                idx_cond = idx[:, -1:]

            logits, past = self._forward_with_cache(idx_cond, past)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
