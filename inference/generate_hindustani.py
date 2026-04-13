from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------
# Repo path
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from sangeet.config import load_yaml, resolve_path
from sangeet.data.dataset import TokenSpec, token_ids_to_codes
from sangeet.data.vocab import load_vocab
from sangeet.model.transformer_lm import CarnaticLMConfig, CarnaticTransformerLM
from sangeet.tokenizer.encodec_codec import (
    EncodecConfig,
    decode_codes_to_wav,
    load_encodec_model,
)
from sangeet.utils.runtime import find_repo_root


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Hindustani audio using trained Transformer + Encodec"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to inference YAML config",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_encode(vocab, key: str) -> int:
    """Return vocab index or unknown."""
    if key in vocab.stoi:
        return vocab.encode(key)
    return vocab.encode("unknown")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@torch.inference_mode()
def main() -> None:

    args = parse_args()
    repo_root = find_repo_root()

    # ------------------------------------------------
    # Load config
    # ------------------------------------------------
    cfg = load_yaml(resolve_path(args.config, base_dir=repo_root))

    ckpt_path = resolve_path(cfg["checkpoint"], base_dir=repo_root)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    run_cfg = ckpt["cfg"]
    token_meta = ckpt["token_meta"]

    # ------------------------------------------------
    # Token spec
    # ------------------------------------------------
    token_spec = TokenSpec(
        n_codebooks=int(token_meta["n_codebooks"]),
        codebook_size=int(token_meta["codebook_size"]),
    )

    # ------------------------------------------------
    # Load vocabs
    # ------------------------------------------------
    vocabs_dir = run_cfg.get("data", {}).get("vocabs_dir")

    if vocabs_dir:
        vocabs_dir = resolve_path(vocabs_dir, base_dir=repo_root)
    else:
        vocabs_dir = ckpt_path.parent.parent / "vocabs"

    vocabs_dir = Path(vocabs_dir)

    print(f"[INFO] Using vocabs: {vocabs_dir}")

    raga_vocab = load_vocab(vocabs_dir / "raga.json")
    tala_vocab = load_vocab(vocabs_dir / "tala.json")
    artist_vocab = load_vocab(vocabs_dir / "artist.json")

    # ------------------------------------------------
    # Build model
    # ------------------------------------------------
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

    model.load_state_dict(ckpt["model"], strict=True)

    # ------------------------------------------------
    # Device
    # ------------------------------------------------
    gen_cfg = cfg.get("generation", {})

    device = gen_cfg.get("device", "cuda")

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    device = torch.device(device)

    model.to(device)
    model.eval()

    print(f"[INFO] Model on: {device}")

    # ------------------------------------------------
    # Conditioning
    # ------------------------------------------------
    cond = cfg.get("conditioning", {})

    raga = str(cond.get("raga", "unknown"))
    tala = str(cond.get("tala", "unknown"))
    artist = str(cond.get("artist", "unknown"))
    text = str(cond.get("text", ""))

    raga_id = _safe_encode(raga_vocab, raga)
    tala_id = _safe_encode(tala_vocab, tala)
    artist_id = _safe_encode(artist_vocab, artist)

    print(f"[INFO] Conditioning:")
    print(f"   Raga   : {raga}")
    print(f"   Tala   : {tala}")
    print(f"   Artist : {artist}")

    # ------------------------------------------------
    # Generation params
    # ------------------------------------------------
    duration_sec = float(gen_cfg.get("duration_sec", 12.0))
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_k = int(gen_cfg.get("top_k", 0))
    top_p = float(gen_cfg.get("top_p", 0.9))

    frame_rate = float(token_meta.get("frame_rate", 50.0))

    n_frames = max(1, int(duration_sec * frame_rate))

    print(f"[INFO] Duration: {duration_sec:.2f}s ({n_frames} frames)")

    # ------------------------------------------------
    # Generate tokens
    # ------------------------------------------------
    print("[INFO] Generating tokens...")

    token_ids = model.generate(
        raga_id=raga_id,
        tala_id=tala_id,
        artist_id=artist_id,
        n_frames=n_frames,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        text=text,
        device=device,
    )

    # ------------------------------------------------
    # Convert → Encodec codes
    # ------------------------------------------------
    codes = token_ids_to_codes(
        token_ids.detach().cpu().numpy().astype(np.int64),
        token_spec,
    )

    # ------------------------------------------------
    # Decode to WAV
    # ------------------------------------------------
    out_cfg = cfg.get("output", {})

    wav_path = resolve_path(
        out_cfg.get("wav_path", "outputs/sample.wav"),
        base_dir=repo_root,
    )

    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    enc_cfg = EncodecConfig(
        model="24khz",
        bandwidth=float(token_meta["encodec_bandwidth"]),
        device=str(device),
        use_normalize=False,
    )

    enc_model = load_encodec_model(enc_cfg)

    print("[INFO] Decoding audio...")

    decode_codes_to_wav(
        enc_model,
        codes=codes,
        out_wav_path=wav_path,
        sample_rate=int(token_meta["encodec_sample_rate"]),
    )

    print(f"[DONE] Audio written to: {wav_path}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
