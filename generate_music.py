"""
generate_music.py

Option B+C: Generate coherent music using 4 codebooks (13s context window)
and stitch multiple clips into a longer piece with crossfades.

Usage:
    python generate_music.py                          # default: 60s output, Kalyān/Tīntāl
    python generate_music.py --raga "Bhairav" --duration 90
    python generate_music.py --list-ragas
    python generate_music.py --list-talas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.audio.postprocess import postprocess_wav
from sangeet.data.dataset import TokenSpec, token_ids_to_codes
from sangeet.data.delay_pattern import delay_tokens_to_codes_v2
from sangeet.data.vocab import load_vocab
from sangeet.model.transformer_lm import CarnaticLMConfig, CarnaticTransformerLM
from sangeet.tokenizer.encodec_codec import (
    EncodecConfig,
    decode_codes_to_wav,
    load_encodec_model,
)
from sangeet.utils.runtime import find_repo_root

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CKPT_PATH       = Path("runs/hindustani_cfg/checkpoints/latest.pt")
OUTPUT_DIR      = Path("outputs_music")

# Codebooks to actually generate — cb0-cb3 cover melody + harmony.
# cb4-cb7 (fine acoustic detail) add static on this model; zero-filled in decoder.
N_CODEBOOKS_USE = 4

# Context-safe clip length: 4096 tokens / 4 codebooks / 75fps = 13.6s
# 12s is the sweet spot — fully within context, no stitching needed.
CLIP_SEC        = 12.0

# Crossfade between clips in seconds (only applies when duration > CLIP_SEC)
CROSSFADE_SEC   = 1.0

# Sampling
TEMPERATURE     = 0.75
TOP_P           = 0.9
CFG_SCALE       = 5.0
CB_TEMPERATURE_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

# Post-processing
POST_HF_CUTOFF_HZ = 10_000.0
POST_TARGET_LUFS  = -14.0
POST_PEAK_DB      = -1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_vocab(path: Path, label: str) -> None:
    import json
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    print(f"\nAvailable {label} ({len(v['itos'])} total):")
    for x in sorted(v["itos"]):
        if not x.startswith("__"):
            print(f"  {x}")


def _safe_encode(vocab, key: str) -> int:
    if key in vocab.stoi:
        return vocab.encode(key)
    # Try case-insensitive match
    for k in vocab.stoi:
        if k.lower() == key.lower():
            return vocab.encode(k)
    print(f"  [WARN] '{key}' not found in vocab — using 'unknown'. Run --list-ragas to see options.")
    return vocab.encode("unknown")


def crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_sec: float) -> np.ndarray:
    """Crossfade two mono/stereo arrays [T] or [T,C]."""
    n = min(int(fade_sec * sr), len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b], axis=0)

    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)

    if a.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in  = fade_in[:, None]

    head   = a[:-n]
    middle = a[-n:] * fade_out + b[:n] * fade_in
    tail   = b[n:]
    return np.concatenate([head, middle, tail], axis=0)


# ---------------------------------------------------------------------------
# Model loading (once, reused across clips)
# ---------------------------------------------------------------------------

def is_delay_pattern_checkpoint(ckpt: dict) -> bool:
    """Returns True if the checkpoint was trained with delay pattern."""
    return bool(ckpt.get("cfg", {}).get("training", {}).get("delay_pattern", False))


def load_model(repo_root: Path, ckpt_path: Path, device: torch.device):
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    model.to(device)
    model.eval()

    return model, token_meta, raga_vocab, tala_vocab, artist_vocab, ckpt


# ---------------------------------------------------------------------------
# Generate one clip
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_clip(
    model: CarnaticTransformerLM,
    token_meta: dict,
    raga_id: int,
    tala_id: int,
    artist_id: int,
    enc_model,
    device: torch.device,
    clip_sec: float,
    n_cb_use: int,
) -> np.ndarray:
    """Returns decoded audio as float32 numpy array [T] or [T,C]."""
    token_spec = model.token_spec
    frame_rate = float(token_meta.get("frame_rate", 75.0))
    n_frames   = max(1, int(clip_sec * frame_rate))
    n_cb_full  = int(token_meta["n_codebooks"])
    sample_rate = int(token_meta["encodec_sample_rate"])

    cb_scales = CB_TEMPERATURE_SCALES[:n_cb_full]

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

    raw_tokens = token_ids.detach().cpu().numpy().astype(np.int64)

    # Decode: delay-pattern checkpoint uses different inverse transform
    if getattr(generate_clip, "_delay_pattern", False):
        codes = delay_tokens_to_codes_v2(
            raw_tokens,
            n_codebooks=n_cb_full,
            codebook_size=int(token_spec.codebook_size),
            token_offset=int(token_spec.token_offset),
        )
        # Skip first K-1 warmup frames (they were PAD-filled)
        codes = codes[:, n_cb_full - 1:]
    else:
        codes = token_ids_to_codes(raw_tokens, token_spec)

    # Zero-fill codebooks beyond n_cb_use
    if n_cb_use < n_cb_full:
        codes[n_cb_use:, :] = 0

    # Decode to temp wav, read back as numpy
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        decode_codes_to_wav(enc_model, codes=codes, out_wav_path=tmp_path, sample_rate=sample_rate)
        audio, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
    finally:
        os.unlink(tmp_path)

    return audio, sr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Hindustani music (Option B+C)")
    p.add_argument("--raga",       default="Kalyāṇ",  help="Raga name (use --list-ragas to see options)")
    p.add_argument("--tala",       default="Tīntāl",  help="Tala name (use --list-talas to see options)")
    p.add_argument("--artist",     default="unknown", help="Artist name")
    p.add_argument("--duration",   type=float, default=12.0, help="Target output duration in seconds (default: 12s — single coherent clip, no stitching)")
    p.add_argument("--clip-sec",   type=float, default=CLIP_SEC, help="Seconds per clip (default: 12)")
    p.add_argument("--n-cb",       type=int,   default=N_CODEBOOKS_USE, help="Codebooks to use 2/4/8 (default: 4)")
    p.add_argument("--cfg-scale",  type=float, default=CFG_SCALE, help="CFG scale (default: 5.0)")
    p.add_argument("--out",        default=None, help="Output wav path (auto-named if omitted)")
    p.add_argument("--ckpt",       default=str(CKPT_PATH), help="Checkpoint path")
    p.add_argument("--list-ragas", action="store_true", help="Print available ragas and exit")
    p.add_argument("--list-talas", action="store_true", help="Print available talas and exit")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    repo_root = find_repo_root()
    ckpt_path = repo_root / args.ckpt

    # -- List vocab and exit --
    vocabs_dir = repo_root / "runs/hindustani_small/vocabs"
    if args.list_ragas:
        list_vocab(vocabs_dir / "raga.json", "ragas")
        return
    if args.list_talas:
        list_vocab(vocabs_dir / "tala.json", "talas")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    model, token_meta, raga_vocab, tala_vocab, artist_vocab, ckpt = load_model(
        repo_root, ckpt_path, device
    )
    use_delay = is_delay_pattern_checkpoint(ckpt)
    generate_clip._delay_pattern = use_delay
    if use_delay:
        print("[INFO] Delay-pattern checkpoint detected — using delay decoding.")

    raga_id   = _safe_encode(raga_vocab,   args.raga)
    tala_id   = _safe_encode(tala_vocab,   args.tala)
    artist_id = _safe_encode(artist_vocab, args.artist)

    print(f"[INFO] Raga: {args.raga} (id={raga_id})  Tala: {args.tala} (id={tala_id})")
    print(f"[INFO] Codebooks: {args.n_cb}/8  |  Clip: {args.clip_sec}s  |  CFG: {args.cfg_scale}")

    enc_cfg   = EncodecConfig(
        model="24khz",
        bandwidth=float(token_meta["encodec_bandwidth"]),
        device=str(device),
        use_normalize=False,
    )
    enc_model = load_encodec_model(enc_cfg)

    # -- How many clips needed --
    n_clips = max(1, int(np.ceil(args.duration / args.clip_sec)))
    print(f"[INFO] Generating {n_clips} clip(s) × {args.clip_sec}s → ~{n_clips * args.clip_sec}s before trim")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clips: list[np.ndarray] = []
    sr_out = None

    for i in range(n_clips):
        print(f"  [{i+1}/{n_clips}] Generating clip {i+1}...", end=" ", flush=True)
        audio, sr = generate_clip(
            model, token_meta,
            raga_id, tala_id, artist_id,
            enc_model, device,
            clip_sec=args.clip_sec,
            n_cb_use=args.n_cb,
        )
        clips.append(audio)
        sr_out = sr
        print(f"done ({len(audio)/sr:.1f}s)")

    # -- Stitch with crossfades --
    print(f"[INFO] Stitching {n_clips} clips with {CROSSFADE_SEC}s crossfade...")
    combined = clips[0]
    for clip in clips[1:]:
        combined = crossfade(combined, clip, sr_out, CROSSFADE_SEC)

    # Trim to requested duration
    max_samples = int(args.duration * sr_out)
    combined    = combined[:max_samples]

    # -- Output path --
    if args.out:
        out_path = Path(args.out)
    else:
        raga_safe = args.raga.replace(" ", "_").replace("/", "-")
        out_path  = OUTPUT_DIR / f"{raga_safe}_{int(args.duration)}s_cb{args.n_cb}.wav"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sr_out, subtype="PCM_16")

    # -- Post-process --
    print(f"[INFO] Post-processing (HF rolloff + LUFS normalisation)...")
    postprocess_wav(
        out_path, out_path,
        hf_cutoff_hz=POST_HF_CUTOFF_HZ,
        target_lufs=POST_TARGET_LUFS,
        peak_db=POST_PEAK_DB,
    )

    print(f"\n[DONE] {out_path}  ({args.duration}s, {args.n_cb} codebooks, cfg={args.cfg_scale})")


if __name__ == "__main__":
    main()
