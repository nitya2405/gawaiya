"""
backend/model_cache.py

Singleton model loader — loads once at worker startup, stays in GPU memory.
Import get_model() from anywhere; the model is loaded only on first call.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch

from backend.config import MODEL_CHECKPOINT, REPO_ROOT, VOCABS_DIR

_lock  = threading.Lock()
_cache: Optional[dict] = None


def get_model() -> dict:
    """
    Returns a dict with keys: model, token_meta, raga_vocab, tala_vocab,
    artist_vocab, enc_model, device.

    Thread-safe: safe to call from multiple threads; loads once.
    """
    global _cache
    if _cache is not None:
        return _cache

    with _lock:
        if _cache is not None:          # double-checked locking
            return _cache

        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from sangeet.data.dataset import TokenSpec
        from sangeet.data.vocab import load_vocab
        from sangeet.model.transformer_lm import CarnaticLMConfig, CarnaticTransformerLM
        from sangeet.tokenizer.encodec_codec import EncodecConfig, load_encodec_model

        ckpt_path = Path(MODEL_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[model_cache] Loading checkpoint {ckpt_path} on {device} …", flush=True)

        ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run_cfg    = ckpt["cfg"]
        token_meta = ckpt["token_meta"]

        token_spec = TokenSpec(
            n_codebooks=int(token_meta["n_codebooks"]),
            codebook_size=int(token_meta["codebook_size"]),
        )

        vocabs_dir = Path(VOCABS_DIR)
        if not vocabs_dir.exists():
            vocabs_dir = ckpt_path.parent.parent / "vocabs"

        raga_vocab   = load_vocab(vocabs_dir / "raga.json")
        tala_vocab   = load_vocab(vocabs_dir / "tala.json")
        artist_vocab = load_vocab(vocabs_dir / "artist.json")

        mcfg = CarnaticLMConfig(
            d_model=int(run_cfg["model"]["d_model"]),
            n_layers=int(run_cfg["model"]["n_layers"]),
            n_heads=int(run_cfg["model"]["n_heads"]),
            dropout=float(run_cfg["model"].get("dropout", 0.1)),
            ff_mult=int(run_cfg["model"].get("ff_mult", 4)),
            cross_attention=bool(run_cfg["model"].get("cross_attention", True)),
            max_seq_len=int(run_cfg["model"].get("max_seq_len", 4096)),
        )

        model = CarnaticTransformerLM(
            mcfg,
            token_spec=token_spec,
            raga_vocab_size=raga_vocab.size,
            tala_vocab_size=tala_vocab.size,
            artist_vocab_size=artist_vocab.size,
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        model.eval()

        enc_cfg = EncodecConfig(
            model="24khz",
            bandwidth=float(token_meta["encodec_bandwidth"]),
            device=str(device),
            use_normalize=False,
        )
        enc_model = load_encodec_model(enc_cfg)

        _cache = {
            "model":        model,
            "token_meta":   token_meta,
            "raga_vocab":   raga_vocab,
            "tala_vocab":   tala_vocab,
            "artist_vocab": artist_vocab,
            "enc_model":    enc_model,
            "device":       device,
        }

        print(f"[model_cache] Ready on {device}.", flush=True)
        return _cache
