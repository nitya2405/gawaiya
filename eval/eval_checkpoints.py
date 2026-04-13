"""
eval_checkpoints.py

Batch evaluation: checkpoints × sampling presets → post-processed WAVs.

Phase-1 improvements active:
  - Per-codebook temperature scaling  (reduces high-cb noise)
  - Temperature annealing             (stability improves toward end of clip)
  - Typical sampling                  (cleaner token distribution)
  - HF rolloff + LUFS normalisation   (post-processing, no retraining)
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
# Checkpoint / preset config
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    "step_70000.pt",
    "step_90000.pt",
    "step_100000.pt",
    "latest.pt",
]

CKPT_DIR = Path("runs/hindustani_small/checkpoints")

PRESETS: dict[str, dict] = {
    "stable": {
        "temperature": 0.6,
        "temperature_anneal_to": 0.5,
        "top_k": 0,
        "top_p": 0.85,
        "typical_mass": 0.9,
        # cfg_scale > 1.0 only makes sense after CFG fine-tuning.
        # Set to 1.0 here for pre-CFG checkpoints; bump to 3.0-5.0 after
        # training with train_hindustani_cfg_finetune.yaml.
        "cfg_scale": 1.0,
    },
    "balanced": {
        "temperature": 0.75,
        "temperature_anneal_to": 0.6,
        "top_k": 0,
        "top_p": 0.9,
        "typical_mass": 0.9,
        "cfg_scale": 1.0,
    },
    "creative": {
        "temperature": 0.9,
        "temperature_anneal_to": 0.7,
        "top_k": 0,
        "top_p": 0.95,
        "typical_mass": 0.95,
        "cfg_scale": 1.0,
    },
    "controlled": {
        "temperature": 0.7,
        "temperature_anneal_to": 0.6,
        "top_k": 100,
        "top_p": 0.0,
        "typical_mass": 0.9,
        "cfg_scale": 1.0,
    },
}

# Per-codebook temperature multipliers.
# cb0-cb1 (melody/harmony) stay at full temperature.
# cb4-cb7 (fine acoustic detail, noise-prone) are dampened.
CB_TEMPERATURE_SCALES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

BASE_CONFIG_PATH = "configs/infer.yaml"
DURATION_SEC = 20.0
CONDITIONING = {
    "raga": "Kalyāṇ",
    "tala": "Tīntāl",
    "artist": "unknown",
    "text": "",
}
OUTPUT_DIR = Path("outputs/v2")

# Post-processing settings
POST_HF_CUTOFF_HZ = 10_000.0   # attenuate Encodec noise above 10 kHz
POST_TARGET_LUFS  = -14.0      # streaming loudness standard
POST_PEAK_DB      = -1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_encode(vocab, key: str) -> int:
    if key in vocab.stoi:
        return vocab.encode(key)
    return vocab.encode("unknown")


def _build_cfg(base_cfg: dict, ckpt_path: Path, preset: dict, out_wav: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["checkpoint"] = str(ckpt_path)
    cfg["conditioning"] = {**CONDITIONING}
    cfg["generation"]["device"] = "cuda"
    cfg["generation"]["duration_sec"] = DURATION_SEC
    cfg["generation"]["temperature"] = preset["temperature"]
    cfg["generation"]["top_k"] = preset["top_k"]
    cfg["generation"]["top_p"] = preset["top_p"]
    cfg["output"]["wav_path"] = str(out_wav)
    return cfg


# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_one(cfg: dict, repo_root: Path, preset: dict) -> None:
    ckpt_path = resolve_path(cfg["checkpoint"], base_dir=repo_root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    run_cfg = ckpt["cfg"]
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
        raise RuntimeError(f"Checkpoint mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}")

    gen_cfg = cfg.get("generation", {})
    device_str = gen_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model.to(device)
    model.eval()

    cond = cfg.get("conditioning", {})
    raga_id   = _safe_encode(raga_vocab,   str(cond.get("raga",   "unknown")))
    tala_id   = _safe_encode(tala_vocab,   str(cond.get("tala",   "unknown")))
    artist_id = _safe_encode(artist_vocab, str(cond.get("artist", "unknown")))
    text      = str(cond.get("text", ""))

    duration_sec = float(gen_cfg.get("duration_sec", 20.0))
    temperature  = float(gen_cfg.get("temperature", 1.0))
    top_k        = int(gen_cfg.get("top_k", 0))
    top_p        = float(gen_cfg.get("top_p", 0.9))

    # Phase-1 / Phase-2 additions from preset
    typical_mass          = float(preset.get("typical_mass", 0.0))
    temperature_anneal_to = preset.get("temperature_anneal_to", None)
    cfg_scale             = float(preset.get("cfg_scale", 1.0))

    # Clamp CB scales to actual number of codebooks in this checkpoint
    n_cb = int(token_meta["n_codebooks"])
    cb_scales = CB_TEMPERATURE_SCALES[:n_cb]
    if len(cb_scales) < n_cb:
        cb_scales = cb_scales + [cb_scales[-1]] * (n_cb - len(cb_scales))

    frame_rate = float(token_meta.get("frame_rate", 50.0))
    n_frames   = max(1, int(duration_sec * frame_rate))

    print(f"         typical_mass={typical_mass}  "
          f"anneal_to={temperature_anneal_to}  "
          f"cfg_scale={cfg_scale}  "
          f"cb_scales={cb_scales}")

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

    out_cfg  = cfg.get("output", {})
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

    decode_codes_to_wav(
        enc_model,
        codes=codes,
        out_wav_path=wav_path,
        sample_rate=int(token_meta["encodec_sample_rate"]),
    )

    # --- Post-processing ---
    print("         [post] HF rolloff + LUFS normalisation...")
    sample_rate = int(token_meta["encodec_sample_rate"])
    postprocess_wav(
        wav_path,
        wav_path,  # in-place
        hf_cutoff_hz=POST_HF_CUTOFF_HZ,
        target_lufs=POST_TARGET_LUFS,
        peak_db=POST_PEAK_DB,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = find_repo_root()
    base_cfg  = load_yaml(resolve_path(BASE_CONFIG_PATH, base_dir=repo_root))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(CHECKPOINTS) * len(PRESETS)
    done  = 0

    for ckpt_name in CHECKPOINTS:
        ckpt_path = repo_root / CKPT_DIR / ckpt_name
        stem = Path(ckpt_name).stem

        for preset_name, preset_params in PRESETS.items():
            done += 1
            out_wav = OUTPUT_DIR / f"{stem}_{preset_name}.wav"

            print(f"\n[{done}/{total}] Generating: {stem} | preset={preset_name}")
            print(
                f"         temp={preset_params['temperature']}  "
                f"top_k={preset_params['top_k']}  "
                f"top_p={preset_params['top_p']}  "
                f"-> {out_wav}"
            )

            try:
                cfg = _build_cfg(base_cfg, ckpt_path, preset_params, out_wav)
                generate_one(cfg, repo_root, preset_params)
                print(f"[OK] Saved: {out_wav}")
            except Exception as exc:
                print(f"[FAILED] {stem} | preset={preset_name} — {type(exc).__name__}: {exc}")

    print(f"\n[DONE] Finished {done} generation(s). Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
