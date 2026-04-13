from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.config import load_yaml, resolve_path
from sangeet.dunya.downloader import (
    crawl_and_build_dataset,
    download_dataset_from_local_metadata,
    get_dunya_token,
)
from sangeet.utils.runtime import find_repo_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Carnatic MP3s from Dunya into a Saraga-style folder structure.")
    p.add_argument("--config", type=str, default="configs/download_carnatic.yaml", help="Path to YAML config.")
    p.add_argument("--mode", type=str, default=None, choices=["local", "crawl"], help="Override config download.mode.")
    p.add_argument("--dataset-root", type=str, default=None, help="Override config dataset.root.")
    p.add_argument("--num-workers", type=int, default=None, help="Override config download.num_workers (local mode).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of recordings (crawl mode).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    cfg = load_yaml(resolve_path(args.config, base_dir=repo_root))

    token_env = cfg["dunya"]["token_env"]
    token = get_dunya_token(token_env)
    try:
        from compmusic import dunya as dn
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `compmusic`. Install project requirements (see README.md)."
        ) from e
    dn.set_token(token)

    dataset_root = cfg["dataset"]["root"]
    if args.dataset_root is not None:
        dataset_root = args.dataset_root
    dataset_root = resolve_path(dataset_root, base_dir=repo_root)

    audio_filename = cfg["dataset"].get("audio_filename", "song.mp3")
    metadata_filename = cfg["dataset"].get("metadata_filename", "song.json")

    mode = cfg["download"]["mode"]
    if args.mode is not None:
        mode = args.mode

    overwrite = bool(args.overwrite or cfg["download"].get("overwrite", False))

    if mode == "local":
        num_workers = cfg["download"].get("num_workers", 4)
        if args.num_workers is not None:
            num_workers = args.num_workers
        download_dataset_from_local_metadata(
            dataset_root,
            audio_filename=audio_filename,
            metadata_filename=metadata_filename,
            overwrite=overwrite,
            num_workers=int(num_workers),
        )
        return

    if mode == "crawl":
        tradition_id = cfg["dunya"]["tradition_id"]
        crawl_and_build_dataset(
            dataset_root,
            tradition_id=tradition_id,
            audio_filename=audio_filename,
            metadata_filename=metadata_filename,
            overwrite=overwrite,
            limit=args.limit,
        )
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
