"""
eval_codebook_depth.py

Diagnostic: generate with different codebook depths to isolate where musical
coherence breaks down.

  n_cb=2 → melody only (cb0+cb1)
  n_cb=4 → melody + harmony
  n_cb=8 → full (current default)

Also tests shorter durations to stay within the trained context window.
Outputs saved to outputs_cb_depth/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.audio.postprocess import postprocess_wav
from sangeet.config import resolve_path
from sangeet.data.dataset import TokenSpec, token_ids_to_codes
from sangeet.data.vocab import load_vocab
from sangeet.model.transformer_lm import CarnaticLMConfig, CarnaticTransformerLM
from sangeet.tokenizer.encodec_codec import (
    EncodecConfig,
    decode_codes_to_wav,
    load_encodec_model,
)
from sangeet.utils.runtime import find_repo_root

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_PATH    = Path("runs/hindustani_cfg/checkpoints/latest.pt")  # step_155000
OUTPUT_DIR   = Path("outputs_cb_depth")

CONDITIONING = {
    "raga":   "Kalyāṇ",   # "Yaman" is not in vocab; Kalyāṇ is the closest match
    "tala":   "Tīntāl",
    "artist": "unknown",
    "text":   "",
}

# (label, duration_sec, n_codebooks_to_use)
# n_codebooks_to_use: generate this many, zero-fill the rest in the decoder
EXPERIMENTS = [
    ("6s_cb2",  6.0,  2),   # melody only, short — well within context
    ("6s_cb4",  6.0,  4),   # melody+harmony, short
    ("6s_cb8",  6.0,  8),   # full, short
    ("10s_cb4", 10.0, 4),   # melody+harmony, medium
    ("10s_cb8", 10.0, 8),   # full, medium
    ("20s_cb4", 20.0, 4),   # melody+harmony, long (current default)
    ("20s_cb8", 20.0, 8),   # full long — baseline (should match cfg_150k)
]

# Sampling — best preset from eval_cfg sweep
TEMPERATURE  = 0.75
TOP_P        = 0.9
CFG_SCALE    = 5.0
# Inverted CB scales: cb0 tight, cb4-cb7 hot
CB_TEMPERATURE_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

POST_HF_CUTOFF_HZ = 10_000.0
POST_TARGET_LUFS  = -14.0
POST_PEAK_DB      = -1.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_encode(vocab, key: str) -> int:
    if key in vocab.stoi:
        return vocab.encode(key)
    return vocab.encode("unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = find_repo_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = repo_root / CKPT_PATH
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    run_cfg    = ckpt["cfg"]
    token_meta = ckpt["token_meta"]

    token_spec = TokenSpec(
        n_codebooks=int(token_meta["n_codebooks"]),
        codebook_size=int(token_meta["codebook_size"]),
    )

    vocabs_dir = Path(run_cfg.get("data", {}).get("vocabs_dir", ""))
    if not vocabs_dir.is_absolute():
        vocabs_dir = repo_root / vocabs_dir
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raga_id   = _safe_encode(raga_vocab,   CONDITIONING["raga"])
    tala_id   = _safe_encode(tala_vocab,   CONDITIONING["tala"])
    artist_id = _safe_encode(artist_vocab, CONDITIONING["artist"])

    frame_rate   = float(token_meta.get("frame_rate", 75.0))
    n_cb_full    = int(token_meta["n_codebooks"])
    cb_size      = int(token_meta["codebook_size"])
    sample_rate  = int(token_meta["encodec_sample_rate"])
    bandwidth    = float(token_meta["encodec_bandwidth"])

    enc_cfg   = EncodecConfig(model="24khz", bandwidth=bandwidth, device=str(device), use_normalize=False)
    enc_model = load_encodec_model(enc_cfg)

    total = len(EXPERIMENTS)
    for i, (label, duration_sec, n_cb_use) in enumerate(EXPERIMENTS, 1):
        out_wav = OUTPUT_DIR / f"{label}.wav"
        n_frames = max(1, int(duration_sec * frame_rate))
        tokens_in_context = n_frames * n_cb_full
        print(
            f"\n[{i}/{total}] {label}  "
            f"({duration_sec}s, {n_frames} frames, {n_cb_use}/{n_cb_full} codebooks, "
            f"{tokens_in_context} tokens in context)"
        )

        cb_scales = CB_TEMPERATURE_SCALES[:n_cb_full]

        with torch.inference_mode():
            token_ids = model.generate(
                raga_id=raga_id,
                tala_id=tala_id,
                artist_id=artist_id,
                n_frames=n_frames,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                cfg_scale=CFG_SCALE,
                cb_temperature_scales=cb_scales,
                device=device,
            )

        # Convert to codes array [n_cb_full, n_frames]
        codes_full = token_ids_to_codes(
            token_ids.detach().cpu().numpy().astype(np.int64),
            token_spec,
        )

        # Zero-fill codebooks beyond n_cb_use
        if n_cb_use < n_cb_full:
            codes_full[n_cb_use:, :] = 0
            print(f"         Zero-filled cb{n_cb_use}–cb{n_cb_full - 1}")

        decode_codes_to_wav(enc_model, codes=codes_full, out_wav_path=out_wav, sample_rate=sample_rate)
        postprocess_wav(out_wav, out_wav, hf_cutoff_hz=POST_HF_CUTOFF_HZ, target_lufs=POST_TARGET_LUFS, peak_db=POST_PEAK_DB)
        print(f"[OK] {out_wav}")

    print(f"\n[DONE]  {total} files in {OUTPUT_DIR}/")
    print("""
Listen in this order:
  6s_cb2  → if this has melody: model works, problem is high codebooks
  6s_cb4  → does adding cb2+cb3 help or hurt?
  6s_cb8  → full model, short — compare to 20s_cb8
  10s_cb4 → does coherence hold over 10s with 4 codebooks?
  20s_cb4 → best candidate for usable output right now
  20s_cb8 → your current 'static' baseline
""")


if __name__ == "__main__":
    main()
