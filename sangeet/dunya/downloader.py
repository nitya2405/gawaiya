from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from sangeet.utils.paths import safe_name


class DunyaAuthError(RuntimeError):
    pass


def get_dunya_token(token_env: str) -> str:
    token = os.environ.get(token_env)
    if not token:
        raise DunyaAuthError(
            f"Missing Dunya API token. Set env var {token_env} (recommended) or update your config."
        )
    return token


@dataclass(frozen=True)
class SongRef:
    mbid: str
    song_dir: Path
    metadata_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_local_songs(
    dataset_root: str | Path,
    *,
    metadata_filename: str = "song.json",
) -> list[SongRef]:
    dataset_root = Path(dataset_root)
    meta_paths = list(dataset_root.rglob(metadata_filename))
    songs: list[SongRef] = []

    for meta_path in meta_paths:
        try:
            data = _read_json(meta_path)
            mbid = str(data.get("mbid", "")).strip()
            if not mbid:
                continue
            songs.append(SongRef(mbid=mbid, song_dir=meta_path.parent, metadata_path=meta_path))
        except Exception:
            continue

    return songs


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.replace(path)


def _retry(
    fn,
    *,
    retries: int = 5,
    initial_backoff_s: float = 1.0,
    max_backoff_s: float = 30.0,
):
    delay = initial_backoff_s
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay = min(max_backoff_s, delay * 2)
    assert last_exc is not None
    raise last_exc


def download_mp3(
    mbid: str,
    *,
    out_path: str | Path,
    overwrite: bool,
) -> None:
    """
    Download an MP3 for a recording MBID into `out_path`.

    Uses PyCompMusic (compmusic) Dunya API. Requires `compmusic.dunya.set_token(...)` to be called.
    """
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return

    def _do() -> bytes:
        from compmusic.dunya import docserver

        return docserver.get_mp3(mbid)

    mp3_bytes: bytes = _retry(_do)
    _atomic_write(out_path, mp3_bytes)


def crawl_carnatic_recordings(
    *,
    tradition_id: str,
    recording_detail: bool = True,
) -> list[dict[str, Any]]:
    from compmusic.dunya import carnatic

    carnatic.set_collections([tradition_id])
    return list(carnatic.get_recordings(recording_detail=recording_detail))


def make_song_dir(
    dataset_root: str | Path,
    *,
    album_title: str | None,
    song_title: str | None,
    mbid: str,
    metadata_filename: str = "song.json",
) -> Path:
    dataset_root = Path(dataset_root)
    album = safe_name(album_title or "unknown_album", max_len=80)
    song = safe_name(song_title or "unknown_song", max_len=80)

    album_dir = dataset_root / album
    song_dir = album_dir / song
    if song_dir.exists():
        # Reuse if this directory already belongs to the same MBID; otherwise avoid collisions.
        meta_path = song_dir / metadata_filename
        if meta_path.exists():
            try:
                existing = _read_json(meta_path)
                if str(existing.get("mbid", "")).strip() == str(mbid).strip():
                    return song_dir
            except Exception:
                pass
        song_dir = album_dir / safe_name(f"{song}-{mbid[:8]}", max_len=90)
    return song_dir


def write_song_metadata(metadata_path: Path, recording_detail: dict[str, Any], *, overwrite: bool) -> None:
    if metadata_path.exists() and not overwrite:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(recording_detail, f, ensure_ascii=False, indent=2)


def download_dataset_from_local_metadata(
    dataset_root: str | Path,
    *,
    audio_filename: str = "song.mp3",
    metadata_filename: str = "song.json",
    overwrite: bool = False,
    num_workers: int = 4,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    songs = discover_local_songs(dataset_root, metadata_filename=metadata_filename)
    if not songs:
        raise FileNotFoundError(f"No {metadata_filename} files found under {Path(dataset_root).resolve()}")

    def task(song: SongRef) -> tuple[str, bool, str | None]:
        out_path = song.song_dir / audio_filename
        try:
            download_mp3(song.mbid, out_path=out_path, overwrite=overwrite)
            return (song.mbid, True, None)
        except Exception as e:
            return (song.mbid, False, str(e))

    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futures = [ex.submit(task, s) for s in songs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading MP3s"):
            mbid, success, err = fut.result()
            if success:
                ok += 1
            else:
                failed += 1
                tqdm.write(f"[FAIL] {mbid}: {err}")

    tqdm.write(f"Done. success={ok} failed={failed} total={len(songs)}")


def crawl_and_build_dataset(
    dataset_root: str | Path,
    *,
    tradition_id: str,
    audio_filename: str = "song.mp3",
    metadata_filename: str = "song.json",
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    recs = crawl_carnatic_recordings(tradition_id=tradition_id, recording_detail=True)
    if limit is not None:
        recs = recs[: max(0, int(limit))]

    for rec in tqdm(recs, desc="Crawling Carnatic recordings"):
        mbid = rec.get("mbid")
        if not mbid:
            continue
        concert = rec.get("concert") or []
        album_title = concert[0].get("title") if concert else None
        song_title = rec.get("title")

        song_dir = make_song_dir(
            dataset_root,
            album_title=album_title,
            song_title=song_title,
            mbid=mbid,
            metadata_filename=metadata_filename,
        )
        meta_path = song_dir / metadata_filename
        mp3_path = song_dir / audio_filename

        write_song_metadata(meta_path, rec, overwrite=overwrite)
        try:
            download_mp3(mbid, out_path=mp3_path, overwrite=overwrite)
        except Exception as e:
            tqdm.write(f"[FAIL] {mbid}: {e}")
