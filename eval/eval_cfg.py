"""
eval_cfg.py

CFG evaluation sweep: checkpoints × presets → post-processed WAVs.
Outputs saved to outputs_cfg_eval/.

Presets include cfg_scale for checkpoints that support classifier-free guidance.
Pre-CFG checkpoints (step_90000, step_100000) are run with cfg_scale=1.0 fallback
if null_cond_emb is missing.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.audio.postprocess import postprocess_wav
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR   = Path("outputs_cfg_eval")
DURATION_SEC = 20.0

# Each entry: (checkpoint_path, label)
CHECKPOINTS = [
    # Pre-CFG baseline — for comparison
    ("runs/hindustani_small/checkpoints/step_120000.pt", "base_120k"),
    # CFG fine-tuned at 150k
    ("runs/hindustani_cfg/checkpoints/latest.pt",        "cfg_150k"),
]

CONDITIONING = {
    "raga":   "Kalyāṇ",   # "Yaman" is not in vocab; falls back to unknown without this fix
    "tala":   "Tīntāl",
    "artist": "unknown",
    "text":   "",
}

PRESETS: dict[str, dict] = {
    # No CFG — apples-to-apples baseline for the base_120k checkpoint
    "no_cfg": {
        "temperature":           0.8,
        "top_p":                 0.9,
        "top_k":                 0,
        "typical_mass":          0.0,
        "temperature_anneal_to": None,
        "cfg_scale":             1.0,
    },
    # CFG scale sweep — applied to cfg_150k; base_120k will auto-fallback to 1.0
    "cfg_3": {
        "temperature":           0.75,
        "top_p":                 0.9,
        "top_k":                 0,
        "typical_mass":          0.0,
        "temperature_anneal_to": None,
        "cfg_scale":             3.0,
    },
    "cfg_4": {
        "temperature":           0.75,
        "top_p":                 0.9,
        "top_k":                 0,
        "typical_mass":          0.0,
        "temperature_anneal_to": None,
        "cfg_scale":             4.0,
    },
    "cfg_5": {
        "temperature":           0.75,
        "top_p":                 0.9,
        "top_k":                 0,
        "typical_mass":          0.0,
        "temperature_anneal_to": None,
        "cfg_scale":             5.0,
    },
}

# Inverted CB scales: cb0 tight (melody coherence), cb4-cb7 hot (escape mode collapse).
# Low temp was confirmed static → high codebooks need MORE temperature, not less.
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


def _has_cfg_support(ckpt: dict) -> bool:
    return "null_cond_emb" in ckpt.get("model", {})


# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_one(
    base_cfg: dict,
    ckpt_path: Path,
    preset: dict,
    out_wav: Path,
    repo_root: Path,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    run_cfg    = ckpt["cfg"]
    token_meta = ckpt["token_meta"]

    token_spec = TokenSpec(
        n_codebooks=int(token_meta["n_codebooks"]),
        codebook_size=int(token_meta["codebook_size"]),
    )

    vocabs_dir = run_cfg.get("data", {}).get("vocabs_dir")
    if vocabs_dir:
        vocabs_dir = resolve_path(vocabs_dir, base_dir=repo_root)
    else:
        vocabs_dir = ckpt_path.parent.parent / "vocabs"
    vocabs_dir = Path(vocabs_dir)

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

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    _expected_new = {"null_cond_emb"}
    bad_missing    = [k for k in missing    if k not in _expected_new]
    bad_unexpected = [k for k in unexpected if k not in _expected_new]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}"
        )

    # If checkpoint doesn't have null_cond_emb, CFG is unsupported → fall back
    cfg_scale = float(preset.get("cfg_scale", 1.0))
    if not _has_cfg_support(ckpt) and cfg_scale > 1.0:
        print(f"         [WARN] Checkpoint has no null_cond_emb — CFG disabled (scale forced to 1.0)")
        cfg_scale = 1.0

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print("         [WARN] CUDA not available, running on CPU.")
    device = torch.device(device_str)

    model.to(device)
    model.eval()

    raga_id   = _safe_encode(raga_vocab,   CONDITIONING["raga"])
    tala_id   = _safe_encode(tala_vocab,   CONDITIONING["tala"])
    artist_id = _safe_encode(artist_vocab, CONDITIONING["artist"])
    text      = CONDITIONING["text"]

    n_cb       = int(token_meta["n_codebooks"])
    cb_scales  = CB_TEMPERATURE_SCALES[:n_cb]
    if len(cb_scales) < n_cb:
        cb_scales = cb_scales + [cb_scales[-1]] * (n_cb - len(cb_scales))

    frame_rate = float(token_meta.get("frame_rate", 50.0))
    n_frames   = max(1, int(DURATION_SEC * frame_rate))

    temperature           = float(preset["temperature"])
    top_k                 = int(preset.get("top_k", 0))
    top_p                 = float(preset.get("top_p", 0.9))
    typical_mass          = float(preset.get("typical_mass", 0.0))
    temperature_anneal_to = preset.get("temperature_anneal_to", None)

    print(
        f"         temp={temperature}  top_k={top_k}  top_p={top_p}  "
        f"typical={typical_mass}  anneal_to={temperature_anneal_to}  cfg={cfg_scale}"
    )

    token_ids = model.generate(
        raga_id=raga_id,
        tala_id=tala_id,
        artist_id=artist_id,
        n_frames=n_frames,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        typical_mass=typical_mass,
        temperature_anneal_to=temperature_anneal_to,
        cb_temperature_scales=cb_scales,
        cfg_scale=cfg_scale,
        text=text,
        device=device,
    )

    codes = token_ids_to_codes(
        token_ids.detach().cpu().numpy().astype(np.int64),
        token_spec,
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    enc_cfg = EncodecConfig(
        model="24khz",
        bandwidth=float(token_meta["encodec_bandwidth"]),
        device=str(device),
        use_normalize=False,
    )
    enc_model = load_encodec_model(enc_cfg)

    decode_codes_to_wav(
        enc_model,
        codes=codes,
        out_wav_path=out_wav,
        sample_rate=int(token_meta["encodec_sample_rate"]),
    )

    print("         [post] HF rolloff + LUFS normalisation...")
    postprocess_wav(
        out_wav,
        out_wav,
        hf_cutoff_hz=POST_HF_CUTOFF_HZ,
        target_lufs=POST_TARGET_LUFS,
        peak_db=POST_PEAK_DB,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = find_repo_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(CHECKPOINTS) * len(PRESETS)
    done  = 0

    for ckpt_rel, label in CHECKPOINTS:
        ckpt_path = repo_root / ckpt_rel

        for preset_name, preset_params in PRESETS.items():
            done   += 1
            out_wav = OUTPUT_DIR / f"{label}_{preset_name}.wav"

            print(
                f"\n[{done}/{total}] Generating: {label} | preset={preset_name} "
                f"| cfg={preset_params['cfg_scale']}"
            )
            print(f"         -> {out_wav}")

            try:
                generate_one(
                    base_cfg  = {},
                    ckpt_path = ckpt_path,
                    preset    = preset_params,
                    out_wav   = out_wav,
                    repo_root = repo_root,
                )
                print(f"[OK] Saved: {out_wav}")
            except Exception as exc:
                print(
                    f"[FAILED] {label} | preset={preset_name} "
                    f"— {type(exc).__name__}: {exc}"
                )

    print(f"\n[DONE] {done} generation(s). Outputs in: {OUTPUT_DIR}/")
    expected = [
        f"{label}_{p}.wav"
        for _, label in CHECKPOINTS
        for p in PRESETS
    ]
    print("\nExpected files:")
    for f in expected:
        print(f"  {OUTPUT_DIR / f}")


if __name__ == "__main__":
    main()
