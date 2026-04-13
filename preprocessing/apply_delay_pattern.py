"""
preprocessing/apply_delay_pattern.py

Re-tokenize all existing segments using the MusicGen delay pattern.

Reads  : data/tokens/hindustani_encodec_24khz_bw6/manifest.jsonl
         + corresponding .npz token files

Writes : data/tokens/hindustani_encodec_24khz_bw6_delay/
           ├── manifest.jsonl     (same fields, updated tokens_path)
           └── <segment_id>.npz   (contains "tokens" array [T*K], int64)

Run with:
    python preprocessing/apply_delay_pattern.py

The resulting manifest is used by train_hindustani_delay.yaml.
Takes ~5-10 minutes on CPU for 15k segments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sangeet.data.delay_pattern import codes_to_delay_tokens_v2
from sangeet.utils.jsonl import read_jsonl


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_MANIFEST = _REPO / "data/tokens/hindustani_encodec_24khz_bw6/manifest.jsonl"
DST_DIR      = _REPO / "data/tokens/hindustani_encodec_24khz_bw6_delay"
CODEBOOK_SIZE = 1024   # Encodec 6kbps
TOKEN_OFFSET  = 2      # PAD=0, BOS=1
PAD_TOKEN_ID  = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-manifest", default=str(SRC_MANIFEST))
    p.add_argument("--dst-dir",      default=str(DST_DIR))
    p.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_manifest = Path(args.src_manifest)
    dst_dir      = Path(args.dst_dir)
    dst_tokens   = dst_dir / "tokens"
    dst_tokens.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(src_manifest))
    print(f"[apply_delay_pattern] {len(rows)} segments → {dst_dir}")

    out_rows = []
    errors   = 0

    for i, row in enumerate(rows):
        src_path = Path(row["tokens_path"])
        if not src_path.is_absolute():
            src_path = (_REPO / src_path).resolve()

        try:
            with np.load(src_path, allow_pickle=False) as z:
                codes = z["codes"]  # [K, T]
        except Exception as e:
            print(f"  [SKIP] {src_path}: {e}")
            errors += 1
            continue

        # Apply delay pattern
        delay_tokens = codes_to_delay_tokens_v2(
            codes,
            codebook_size=args.codebook_size,
            pad_token_id=PAD_TOKEN_ID,
            token_offset=TOKEN_OFFSET,
        )

        # Save
        stem     = Path(src_path).stem
        dst_path = dst_tokens / f"{stem}.npz"
        np.savez_compressed(dst_path, tokens=delay_tokens)

        new_row = dict(row)
        new_row["tokens_path"] = str(dst_path.relative_to(_REPO))
        new_row["delay_pattern"] = True
        out_rows.append(new_row)

        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            print(f"  [{i+1}/{len(rows)}] done  ({errors} errors)")

    # Write new manifest
    manifest_out = dst_dir / "manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[DONE] {len(out_rows)} segments written to {dst_dir}")
    print(f"       manifest → {manifest_out}")
    if errors:
        print(f"       {errors} segments skipped (load errors)")


if __name__ == "__main__":
    main()
