from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.config import load_yaml, resolve_path
from sangeet.tokenizer.encodec_codec import decode_codes_to_wav, encode_wav_file, load_encodec_model
from sangeet.tokenizer.encodec_codec import EncodecConfig
from sangeet.utils.jsonl import read_jsonl
from sangeet.utils.runtime import find_repo_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encodec tokenization for Carnatic WAV segments.")
    p.add_argument("--config", type=str, default="configs/tokenize_encodec_24khz.yaml", help="Path to YAML config.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing token files.")
    p.add_argument("--decode-sanity", action="store_true", help="Decode the first token file back to WAV.")
    return p.parse_args()


def _maybe_rel(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _resolve_path(p: str | Path, *, repo_root: Path, manifest_dir: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    cand = (repo_root / path).resolve()
    if cand.exists():
        return cand
    return (manifest_dir / path).resolve()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    cfg = load_yaml(resolve_path(args.config, base_dir=repo_root))

    input_manifest = resolve_path(cfg["input_manifest"], base_dir=repo_root)
    tokens_dir = resolve_path(cfg["output"]["tokens_dir"], base_dir=repo_root)
    out_manifest = resolve_path(cfg["output"]["manifest_path"], base_dir=repo_root)
    manifest_dir = Path(input_manifest).resolve().parent

    tcfg = cfg["tokenizer"]
    enc_cfg = EncodecConfig(
        model=str(tcfg.get("model", "24khz")),
        bandwidth=float(tcfg.get("bandwidth", 6.0)),
        device=str(tcfg.get("device", "cuda")),
        use_normalize=bool(tcfg.get("use_normalize", False)),
    )

    model = load_encodec_model(enc_cfg)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    first_tokens_path: Path | None = None

    with out_manifest.open("w", encoding="utf-8") as f_out:
        for row in tqdm(read_jsonl(input_manifest), desc="Tokenizing"):
            mbid = str(row.get("mbid", "")).strip()
            seg_idx = int(row.get("segment_index", 0))
            wav_path = _resolve_path(str(row["wav_path"]), repo_root=repo_root, manifest_dir=manifest_dir)

            tok_path = tokens_dir / mbid / f"seg_{seg_idx:05d}.npz"
            tok_path.parent.mkdir(parents=True, exist_ok=True)

            if (not tok_path.exists()) or args.overwrite:
                #enc = encode_wav_file(model, wav_path)


                try:
                    enc = encode_wav_file(model, wav_path)
                except Exception as e:
                    print(f"[SKIP] {wav_path}: {e}")
                    continue

                codes = enc["codes"].astype(np.int16)
                np.savez_compressed(
                    tok_path,
                    codes=codes,
                    sample_rate=np.array([enc["sample_rate"]], dtype=np.int32),
                    channels=np.array([enc["channels"]], dtype=np.int32),
                    bandwidth=np.array([enc["bandwidth"]], dtype=np.float32),
                    n_codebooks=np.array([enc["n_codebooks"]], dtype=np.int32),
                    codebook_size=np.array([enc["codebook_size"]], dtype=np.int32),
                    frame_rate=np.array([enc["frame_rate"]], dtype=np.float32),
                )
            else:
                with np.load(tok_path, allow_pickle=False) as z:
                    codes = z["codes"]

                        

                    enc = {
                        "sample_rate": int(z["sample_rate"][0]),
                        "channels": int(z["channels"][0]),
                        "bandwidth": float(z["bandwidth"][0]),
                        "n_codebooks": int(z["n_codebooks"][0]),
                        "codebook_size": int(z["codebook_size"][0]),
                        "frame_rate": float(z["frame_rate"][0]),
                    }

            if first_tokens_path is None:
                first_tokens_path = tok_path

            n_codebooks, n_frames = int(codes.shape[0]), int(codes.shape[1])

            out_row: dict[str, Any] = dict(row)
            out_row.update(
                {
                    "tokens_path": _maybe_rel(tok_path, base=repo_root),
                    "n_codebooks": n_codebooks,
                    "n_frames": n_frames,
                    "codebook_size": int(enc["codebook_size"]),
                    "frame_rate": float(enc["frame_rate"]),
                    "encodec_sample_rate": int(enc["sample_rate"]),
                    "encodec_bandwidth": float(enc["bandwidth"]),
                }
            )
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote tokens manifest: {out_manifest}")

    if args.decode_sanity and first_tokens_path is not None:
        with np.load(first_tokens_path, allow_pickle=False) as z:
            codes = z["codes"]
            decode_out = out_manifest.parent / "decode_sanity.wav"
            decode_codes_to_wav(model, codes=codes, out_wav_path=decode_out)
            print(f"Decoded sanity check WAV: {decode_out}")


if __name__ == "__main__":
    main()
