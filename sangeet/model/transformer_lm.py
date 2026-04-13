from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sangeet.data.dataset import TokenSpec
from sangeet.model.rope import RotaryEmbedding, apply_rope
from sangeet.utils.text import ByteTokenizer


@dataclass(frozen=True)
class CarnaticLMConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.1
    ff_mult: int = 4
    cross_attention: bool = True
    rope_theta: float = 10000.0
    max_seq_len: int = 16384
    # Classifier-free guidance: probability of dropping ALL conditioning
    # during training.  0.0 = disabled.  Recommended: 0.1.
    cfg_dropout: float = 0.0


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, dropout: float, rope_theta: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.dropout = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, theta=float(rope_theta))

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_offset: int,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        sin, cos = self.rope.get_sin_cos(t, device=x.device, dtype=x.dtype, offset=int(pos_offset))
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        if past_kv is not None:
            past_k, past_v = past_kv
            # In generation we append one token at a time.
            if t != 1:
                raise ValueError("Caching path expects t==1")
            # Fast path: preallocated KV cache (avoid concat each step).
            if use_cache and past_k.ndim == 4 and past_k.shape[2] > int(pos_offset):
                past_k[:, :, int(pos_offset) : int(pos_offset) + 1, :] = k
                past_v[:, :, int(pos_offset) : int(pos_offset) + 1, :] = v
                k = past_k[:, :, : int(pos_offset) + 1, :]
                v = past_v[:, :, : int(pos_offset) + 1, :]
                new_kv = (past_k, past_v)
            else:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                new_kv = (k, v) if use_cache else None
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=(self.dropout if self.training else 0.0), is_causal=False
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=(self.dropout if self.training else 0.0), is_causal=True
            )
            new_kv = (k, v) if use_cache else None

        out = attn.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out(out)
        return out, new_kv


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.dropout = float(dropout)

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, *, mem_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, _ = x.shape
        s = mem.shape[1]

        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Hd]
        kv = self.kv(mem)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,S,Hd]
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mem_mask is not None:
            # mem_mask: [B,S] with True for valid positions.
            attn_mask = mem_mask[:, None, None, :].to(torch.bool)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=False,
        )
        out = attn.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mult)
        self.in_proj = nn.Linear(d_model, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.out_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: CarnaticLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = SelfAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, rope_theta=cfg.rope_theta
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout) if cfg.cross_attention else None
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, mult=cfg.ff_mult, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mem: torch.Tensor | None,
        mem_mask: torch.Tensor | None,
        pos_offset: int,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_kv = self.self_attn(self.ln1(x), pos_offset=pos_offset, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        if self.cross_attn is not None and mem is not None:
            x = x + self.cross_attn(self.ln2(x), mem, mem_mask=mem_mask)
        x = x + self.ff(self.ln3(x))
        return x, new_kv


class CarnaticTransformerLM(nn.Module):
    def __init__(
        self,
        cfg: CarnaticLMConfig,
        *,
        token_spec: TokenSpec,
        raga_vocab_size: int,
        tala_vocab_size: int,
        artist_vocab_size: int,
        text_vocab_size: int = ByteTokenizer().vocab_size,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_spec = token_spec

        self.embed_tokens = nn.Embedding(
            token_spec.vocab_size, cfg.d_model, padding_idx=int(token_spec.pad_id)
        )
        self.drop = nn.Dropout(cfg.dropout)

        self.raga_emb = nn.Embedding(int(raga_vocab_size), cfg.d_model)
        self.tala_emb = nn.Embedding(int(tala_vocab_size), cfg.d_model)
        self.artist_emb = nn.Embedding(int(artist_vocab_size), cfg.d_model)
        self.text_emb = nn.Embedding(int(text_vocab_size), cfg.d_model)
        self.cond_type_emb = nn.Embedding(4, cfg.d_model)  # raga,tala,artist,text

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(int(cfg.n_layers))])
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Per-codebook heads over codebook_size.
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, int(token_spec.codebook_size), bias=False) for _ in range(int(token_spec.n_codebooks))]
        )

        # Learned null/unconditional embedding used for classifier-free guidance.
        # Replaces all conditioning slots when cfg_dropout fires during training
        # or when running the unconditional forward pass at inference.
        self.null_cond_emb = nn.Parameter(torch.zeros(cfg.d_model))

    def build_memory(
        self,
        *,
        raga_id: torch.Tensor,
        tala_id: torch.Tensor,
        artist_id: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        use_uncond: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the conditioning memory matrix [B, S, D] for cross-attention.

        use_uncond=True: replace ALL conditioning with the learned null embedding.
            Used for the unconditional forward pass in CFG inference.

        During training, if cfg.cfg_dropout > 0, each sample in the batch
        independently drops all conditioning with that probability (i.e. the
        model learns both conditional and unconditional distributions).
        """
        b = int(raga_id.shape[0])

        if use_uncond:
            # Full unconditional pass: every slot gets the null embedding.
            null = self.null_cond_emb.view(1, 1, -1).expand(b, 3, -1)
            mem = null.clone()
            mem_mask = torch.ones((b, 3), device=mem.device, dtype=torch.bool)
            return mem, mem_mask

        r = self.raga_emb(raga_id) + self.cond_type_emb.weight[0]   # [B, D]
        t = self.tala_emb(tala_id) + self.cond_type_emb.weight[1]
        a = self.artist_emb(artist_id) + self.cond_type_emb.weight[2]

        # --- CFG conditioning dropout (training only) ---
        if self.training and self.cfg.cfg_dropout > 0.0:
            # Drop ALL conditions for a random subset of the batch.
            drop = torch.rand(b, device=raga_id.device) < self.cfg.cfg_dropout
            null = self.null_cond_emb.view(1, -1).expand(b, -1)  # [B, D]
            r = torch.where(drop.unsqueeze(-1), null, r)
            t = torch.where(drop.unsqueeze(-1), null, t)
            a = torch.where(drop.unsqueeze(-1), null, a)

        mem = torch.stack([r, t, a], dim=1)  # [B, 3, D]
        mem_mask = torch.ones((b, mem.shape[1]), device=mem.device, dtype=torch.bool)

        if text_ids is not None:
            te = self.text_emb(text_ids) + self.cond_type_emb.weight[3]
            tmask = text_ids != ByteTokenizer().pad_id
            mem = torch.cat([mem, te], dim=1)
            mem_mask = torch.cat([mem_mask, tmask], dim=1)

        return mem, mem_mask

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        raga_id: torch.Tensor,
        tala_id: torch.Tensor,
        artist_id: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
        use_uncond: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        """
        Returns hidden states [B,T,D]. If use_cache=True, returns updated kv cache per layer.
        use_uncond=True routes to null conditioning for the CFG unconditional pass.
        """
        x = self.embed_tokens(input_ids)
        x = self.drop(x)

        mem, mem_mask = self.build_memory(
            raga_id=raga_id, tala_id=tala_id, artist_id=artist_id,
            text_ids=text_ids, use_uncond=use_uncond,
        )

        new_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = [] if use_cache else None
        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        # Absolute positions for this forward call start at pos_offset.
        for i, block in enumerate(self.blocks):
            x, kv = block(
                x,
                mem=mem if self.cfg.cross_attention else None,
                mem_mask=mem_mask if self.cfg.cross_attention else None,
                pos_offset=int(pos_offset),
                past_kv=past_kv[i],
                use_cache=use_cache,
            )
            if use_cache and new_cache is not None:
                new_cache.append(kv)

        x = self.final_ln(x)
        return x, new_cache

    def compute_logits(self, hidden: torch.Tensor, *, pos_offset: int) -> torch.Tensor:
        """
        Map hidden states [B,T,D] to logits [B,T,codebook_size] using the appropriate codebook head per position.
        """
        # Ensure FP32 for stable logits under AMP
        hidden = hidden.float()

        b, t, _ = hidden.shape

        k = int(self.token_spec.n_codebooks)
        cb = int(self.token_spec.codebook_size)

        logits = hidden.new_empty((b, t, cb))
        pos = torch.arange(t, device=hidden.device, dtype=torch.long)
        head_idx = (pos + int(pos_offset)) % k
        for i in range(k):
            m = head_idx == i
            if torch.any(m):
                #logits[:, m, :] = self.heads[i](hidden[:, m, :])
                logits[:, m, :] = self.heads[i](hidden[:, m, :]).float()

        return logits

    def targets_to_codes(self, target_ids: torch.Tensor, *, pos_offset: int) -> torch.Tensor:
        """
        Convert target token ids [B,T] into code values [B,T] in 0..codebook_size-1.
        PAD positions become -100 (ignore_index for CE loss).
        """
        pad_id = int(self.token_spec.pad_id)
        k = int(self.token_spec.n_codebooks)
        cb = int(self.token_spec.codebook_size)
        off = int(self.token_spec.token_offset)

        b, t = target_ids.shape
        pos = torch.arange(t, device=target_ids.device, dtype=torch.long)
        head_idx = (pos + int(pos_offset)) % k  # [T]
        offsets = off + head_idx * cb  # [T]
        codes = target_ids - offsets.unsqueeze(0)
        codes = codes.to(torch.long)
        codes = torch.where(target_ids == pad_id, torch.full_like(codes, -100), codes)
        return codes

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None,
        raga_id: torch.Tensor,
        tala_id: torch.Tensor,
        artist_id: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        cb_loss_weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, _ = self.forward_hidden(
            input_ids,
            raga_id=raga_id,
            tala_id=tala_id,
            artist_id=artist_id,
            text_ids=text_ids,
            past_kv=None,
            pos_offset=0,
            use_cache=False,
        )
        logits = self.compute_logits(hidden, pos_offset=0)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if target_ids is not None:
            target_codes = self.targets_to_codes(target_ids, pos_offset=0)

            if cb_loss_weights is not None:
                # Weighted cross-entropy: upweight early codebooks (melody)
                # and downweight late ones (fine acoustic detail / noise-prone).
                k  = int(self.token_spec.n_codebooks)
                cb = int(self.token_spec.codebook_size)
                b, t = target_codes.shape
                pos      = torch.arange(t, device=logits.device)
                head_idx = pos % k  # [T]

                weights_t = logits.new_tensor(cb_loss_weights[:k])
                total_loss   = logits.new_zeros(())
                total_weight = 0.0

                for i in range(k):
                    mask = head_idx == i
                    if not mask.any():
                        continue
                    cb_logits  = logits[:, mask, :].reshape(-1, cb)
                    cb_targets = target_codes[:, mask].reshape(-1)
                    cb_loss    = F.cross_entropy(cb_logits, cb_targets, ignore_index=-100)
                    total_loss   = total_loss + weights_t[i] * cb_loss
                    total_weight += float(weights_t[i].item())

                loss = total_loss / max(total_weight, 1e-8)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_codes.reshape(-1),
                    ignore_index=-100,
                )

            out["loss"] = loss
        return out

    @torch.inference_mode()
    def generate(
        self,
        *,
        raga_id: int,
        tala_id: int,
        artist_id: int,
        n_frames: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        typical_mass: float = 0.0,
        temperature_anneal_to: float | None = None,
        cb_temperature_scales: list[float] | None = None,
        cfg_scale: float = 1.0,
        text: str = "",
        max_text_len: int = 128,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """
        Generate flattened token ids [T*K] (without BOS) for `n_frames`.

        Args:
            temperature:            Base sampling temperature.
            temperature_anneal_to:  If set, linearly anneal temperature from
                                    `temperature` down to this value over n_frames.
            cb_temperature_scales:  Per-codebook temperature multipliers, length
                                    n_codebooks.  E.g. [1.0, 0.95, 0.9, 0.85,
                                    0.8, 0.75, 0.7, 0.65] reduces noise from
                                    high-frequency codebooks.
            typical_mass:           Typical-sampling probability mass (0 = off).
                                    ~0.9 works well.
            top_k / top_p:          Standard nucleus / top-k filters.
            cfg_scale:              Classifier-free guidance scale.  1.0 = off.
                                    3.0–5.0 recommended for trained CFG models.
                                    Runs a second unconditional forward pass each
                                    step:  logits = uncond + scale*(cond - uncond).
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        tok = ByteTokenizer()
        text_ids = None
        if text:
            ids = tok.encode(text, max_len=int(max_text_len))
            text_ids = torch.tensor([ids], device=dev, dtype=torch.long)

        raga_t = torch.tensor([int(raga_id)], device=dev, dtype=torch.long)
        tala_t = torch.tensor([int(tala_id)], device=dev, dtype=torch.long)
        artist_t = torch.tensor([int(artist_id)], device=dev, dtype=torch.long)

        k = int(self.token_spec.n_codebooks)
        cb = int(self.token_spec.codebook_size)
        off = int(self.token_spec.token_offset)
        total_tokens = int(n_frames) * k

        current = torch.tensor([[int(self.token_spec.bos_id)]], device=dev, dtype=torch.long)

        # --- Pre-allocate KV caches ---
        head_dim    = int(self.cfg.d_model // self.cfg.n_heads)
        cache_dtype = self.embed_tokens.weight.dtype
        use_cfg     = cfg_scale > 1.0

        def _make_kv_cache() -> list[tuple[torch.Tensor, torch.Tensor]]:
            c = []
            for _ in range(len(self.blocks)):
                kc = torch.empty((1, int(self.cfg.n_heads), total_tokens, head_dim), device=dev, dtype=cache_dtype)
                vc = torch.empty((1, int(self.cfg.n_heads), total_tokens, head_dim), device=dev, dtype=cache_dtype)
                c.append((kc, vc))
            return c

        cache_cond  = _make_kv_cache()
        cache_uncond = _make_kv_cache() if use_cfg else None

        input_pos = 0
        anneal_to = float(temperature_anneal_to) if temperature_anneal_to is not None else None

        out_tokens: list[int] = []
        while len(out_tokens) < total_tokens:
            # --- Conditioned forward pass ---
            hidden_cond, cache_cond = self.forward_hidden(
                current,
                raga_id=raga_t,
                tala_id=tala_t,
                artist_id=artist_t,
                text_ids=text_ids,
                past_kv=cache_cond,
                pos_offset=int(input_pos),
                use_cache=True,
                use_uncond=False,
            )
            logits = self.compute_logits(hidden_cond, pos_offset=int(input_pos))  # [1,1,cb]

            # --- CFG: blend with unconditional logits ---
            if use_cfg:
                hidden_uncond, cache_uncond = self.forward_hidden(
                    current,
                    raga_id=raga_t,
                    tala_id=tala_t,
                    artist_id=artist_t,
                    text_ids=text_ids,
                    past_kv=cache_uncond,
                    pos_offset=int(input_pos),
                    use_cache=True,
                    use_uncond=True,
                )
                logits_uncond = self.compute_logits(hidden_uncond, pos_offset=int(input_pos))
                # Standard CFG: push logits away from unconditional distribution
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            # --- Temperature annealing (frame-level) ---
            frame_idx = input_pos // k
            if anneal_to is not None:
                progress  = min(1.0, frame_idx / max(1, n_frames - 1))
                base_temp = temperature + (anneal_to - temperature) * progress
            else:
                base_temp = temperature

            # --- Per-codebook temperature scale ---
            codebook_idx = int(input_pos % k)
            if cb_temperature_scales is not None:
                effective_temp = base_temp * float(cb_temperature_scales[codebook_idx])
            else:
                effective_temp = base_temp
            effective_temp = max(effective_temp, 1e-6)

            next_code = sample_from_logits(
                logits[0, 0],
                temperature=effective_temp,
                top_k=top_k,
                top_p=top_p,
                typical_mass=typical_mass,
            )

            next_token = int(off + codebook_idx * cb + int(next_code))
            out_tokens.append(next_token)

            current = torch.tensor([[next_token]], device=dev, dtype=torch.long)
            input_pos += 1

        return torch.tensor(out_tokens, device=dev, dtype=torch.long)


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    typical_mass: float = 0.0,
) -> int:
    """
    Sample one token index from logits.

    Filtering order: temperature → typical → top-k → top-p.

    typical_mass: probability mass to retain using typical sampling
                  (Meister et al. 2023).  0 = disabled.  ~0.9 recommended.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits.float() / float(temperature)

    # --- Typical sampling ---
    # Keeps tokens whose surprisal (-log p) is closest to the conditional
    # entropy H of the distribution.  Removes both over-confident (boring)
    # and over-surprising (noisy) tokens.
    if typical_mass > 0.0:
        probs = torch.softmax(logits, dim=-1)
        neg_log_p = -torch.log(probs + 1e-8)
        entropy = (probs * neg_log_p).sum()
        diff = torch.abs(neg_log_p - entropy)
        sorted_diff, sorted_idx = torch.sort(diff)
        sorted_probs = probs[sorted_idx]
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= float(typical_mass)
        keep[0] = True  # always keep the most typical token
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[sorted_idx[keep]] = True
        logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

    # --- Top-k ---
    if top_k and top_k > 0:
        top_k = int(top_k)
        v, ix = torch.topk(logits, k=min(top_k, logits.numel()))
        mask = torch.full_like(logits, float("-inf"))
        mask[ix] = logits[ix]
        logits = mask

    # --- Top-p (nucleus) ---
    if top_p and top_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= float(top_p)
        keep[..., 0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[sorted_idx[keep]] = True
        logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return int(idx)
