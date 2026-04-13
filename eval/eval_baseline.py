"""
eval_baseline.py

Clean diagnostic generation from the base model.
No CFG, no typical sampling — isolates model quality from sampling tricks.

Run WHILE CFG training is in progress to confirm the base model sounds OK.
"""

from __future__ import annotations
import copy, sys
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
from sangeet.tokenizer.encodec_codec import EncodecConfig, decode_codes_to_wav, load_encodec_model
from sangeet.utils.runtime import find_repo_root

# ---------------------------------------------------------------------------
# Settings — deliberately minimal to isolate model quality
# ---------------------------------------------------------------------------

# Best base checkpoint (no CFG)
CHECKPOINT = "runs/hindustani_small/checkpoints/step_120000.pt"

CONDITIONING = {"raga": "Kalyāṇ", "tala": "Tīntāl", "artist": "unknown", "text": ""}
DURATION_SEC = 20.0
OUTPUT_DIR   = Path("outputs_baseline")

# Minimal presets — no typical sampling, no cb scaling, no CFG
# Now testing with INVERTED cb scales: cb0 tight, cb4-cb7 hot
# Overall temp can be lower now since high cbs get boosted automatically
PRESETS = {
    "inv_t065": {"temperature": 0.65, "top_p": 0.9,  "top_k": 0, "typical_mass": 0.0, "cfg_scale": 1.0},
    "inv_t075": {"temperature": 0.75, "top_p": 0.9,  "top_k": 0, "typical_mass": 0.0, "cfg_scale": 1.0},
    "inv_t085": {"temperature": 0.85, "top_p": 0.92, "top_k": 0, "typical_mass": 0.0, "cfg_scale": 1.0},
    # Also test with top-k on high codebooks (applied globally here, but still useful)
    "inv_t075_topk50": {"temperature": 0.75, "top_p": 0.0, "top_k": 50, "typical_mass": 0.0, "cfg_scale": 1.0},
}

POST_HF_CUTOFF_HZ = 10_000.0
POST_TARGET_LUFS  = -14.0
POST_PEAK_DB      = -1.0

# Inverted from original: cb0 stays tight (melody coherence),
# cb4-cb7 pushed hotter to escape mode collapse in texture codebooks.
CB_TEMPERATURE_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def _safe_encode(vocab, key: str) -> int:
    return vocab.encode(key) if key in vocab.stoi else vocab.encode("unknown")


@torch.inference_mode()
def generate_one(ckpt_path: Path, preset: dict, out_wav: Path, repo_root: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_cfg    = ckpt["cfg"]
    token_meta = ckpt["token_meta"]

    token_spec = TokenSpec(
        n_codebooks=int(token_meta["n_codebooks"]),
        codebook_size=int(token_meta["codebook_size"]),
    )

    vocabs_dir = run_cfg.get("data", {}).get("vocabs_dir")
    vocabs_dir = Path(resolve_path(vocabs_dir, base_dir=repo_root) if vocabs_dir
                      else ckpt_path.parent.parent / "vocabs")

    raga_vocab   = load_vocab(vocabs_dir / "raga.json")
    tala_vocab   = load_vocab(vocabs_dir / "tala.json")
    artist_vocab = load_vocab(vocabs_dir / "artist.json")

    mcfg = CarnaticLMConfig(
        d_model    = int(run_cfg["model"]["d_model"]),
        n_layers   = int(run_cfg["model"]["n_layers"]),
        n_heads    = int(run_cfg["model"]["n_heads"]),
        dropout    = float(run_cfg["model"].get("dropout", 0.1)),
        ff_mult    = int(run_cfg["model"].get("ff_mult", 4)),
        cross_attention = bool(run_cfg["model"].get("cross_attention", True)),
        max_seq_len     = int(run_cfg["model"].get("max_seq_len", 4096)),
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
    model.to(device).eval()

    raga_id   = _safe_encode(raga_vocab,   CONDITIONING["raga"])
    tala_id   = _safe_encode(tala_vocab,   CONDITIONING["tala"])
    artist_id = _safe_encode(artist_vocab, CONDITIONING["artist"])

    n_cb      = int(token_meta["n_codebooks"])
    cb_scales = CB_TEMPERATURE_SCALES[:n_cb]

    frame_rate = float(token_meta.get("frame_rate", 50.0))
    n_frames   = max(1, int(DURATION_SEC * frame_rate))

    token_ids = model.generate(
        raga_id=raga_id, tala_id=tala_id, artist_id=artist_id,
        n_frames=n_frames,
        temperature=float(preset["temperature"]),
        top_k=int(preset.get("top_k", 0)),
        top_p=float(preset.get("top_p", 0.0)),
        typical_mass=float(preset.get("typical_mass", 0.0)),
        temperature_anneal_to=None,       # disabled for clean diagnostic
        cb_temperature_scales=cb_scales,  # keep cb dampening (this is safe)
        cfg_scale=1.0,                    # always off for base model
        device=device,
    )

    codes = token_ids_to_codes(
        token_ids.detach().cpu().numpy().astype(np.int64), token_spec
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    enc_cfg = EncodecConfig(
        model="24khz",
        bandwidth=float(token_meta["encodec_bandwidth"]),
        device=str(device),
        use_normalize=False,
    )
    decode_codes_to_wav(load_encodec_model(enc_cfg), codes=codes,
                        out_wav_path=out_wav,
                        sample_rate=int(token_meta["encodec_sample_rate"]))

    postprocess_wav(out_wav, out_wav,
                    hf_cutoff_hz=POST_HF_CUTOFF_HZ,
                    target_lufs=POST_TARGET_LUFS,
                    peak_db=POST_PEAK_DB)


def main() -> None:
    repo_root = find_repo_root()
    ckpt_path = repo_root / CHECKPOINT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(PRESETS)
    for i, (name, params) in enumerate(PRESETS.items(), 1):
        out_wav = OUTPUT_DIR / f"step120000_{name}.wav"
        print(f"\n[{i}/{total}] {name}  temp={params['temperature']}  "
              f"top_p={params['top_p']}  typical={params['typical_mass']}  -> {out_wav}")
        try:
            generate_one(ckpt_path, params, out_wav, repo_root)
            print(f"[OK] {out_wav}")
        except Exception as e:
            print(f"[FAILED] {name} — {type(e).__name__}: {e}")

    print(f"\n[DONE]  Listen order: inv_t065 → inv_t075 → inv_t085 → inv_t075_topk50")
    print("  Goal: find lowest temperature that still sounds musical (not static).")
    print("  cb0 is now held tight for melody; cb4-cb7 run hot to avoid mode collapse.")


if __name__ == "__main__":
    main()
