from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.audio.ffmpeg import decode_segment_to_wav, ffprobe_duration_sec
from sangeet.audio.normalize import NormalizeConfig, normalize_audio
from sangeet.config import load_yaml, resolve_path
from sangeet.utils.runtime import find_repo_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MP3 -> WAV preprocessing (resample/normalize/segment) + manifest.")
    p.add_argument("--config", type=str, default="configs/preprocess.yaml", help="Path to YAML config.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV segments.")
    p.add_argument("--max-songs", type=int, default=None, help="Process at most N songs (debug).")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_rel(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _get_first_name(items: Any) -> str | None:
    if not items:
        return None
    if isinstance(items, list) and items:
        it = items[0]
        if isinstance(it, dict):
            name = it.get("name") or it.get("common_name") or it.get("title")
            if name:
                return str(name)
    return None


def extract_conditioning(meta: dict[str, Any]) -> tuple[str, str, str]:
    # Raga / Raaga
    raga = (
        _get_first_name(meta.get("raaga"))
        or _get_first_name(meta.get("raga"))
        or _get_first_name(meta.get("raags"))
        or "unknown"
    )
    # Tala / Taala
    tala = _get_first_name(meta.get("taala")) or _get_first_name(meta.get("taals")) or "unknown"

    # Artist
    artist = "unknown"
    artists = meta.get("artists") or []
    if isinstance(artists, list) and artists:
        lead = None
        for a in artists:
            if isinstance(a, dict) and a.get("lead"):
                lead = a
                break
        pick = lead or artists[0]
        if isinstance(pick, dict):
            a = pick.get("artist") or {}
            if isinstance(a, dict) and a.get("name"):
                artist = str(a["name"])

    if artist == "unknown":
        album_artists = meta.get("album_artists") or []
        if isinstance(album_artists, list) and album_artists:
            aa = album_artists[0]
            if isinstance(aa, dict) and aa.get("name"):
                artist = str(aa["name"])

    return str(raga), str(tala), str(artist)


def _get_duration_sec(meta: dict[str, Any], *, audio_path: Path) -> float:
    # Saraga/Dunya metadata uses milliseconds for `length` (even for short clips),
    # so treat it as ms unconditionally when present.
    v = meta.get("length")
    if isinstance(v, (int, float)) and v > 0:
        return float(v) / 1000.0

    for k in ("duration_ms", "duration"):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v > 0:
            if k.endswith("_ms"):
                return float(v) / 1000.0
            # Heuristic: if duration is very large, assume milliseconds.
            if float(v) > 60 * 60:  # > 1 hour as seconds is unlikely for a single track
                return float(v) / 1000.0
            return float(v)
    return ffprobe_duration_sec(audio_path)


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    cfg = load_yaml(resolve_path(args.config, base_dir=repo_root))

    dataset_root = resolve_path(cfg["dataset"]["root"], base_dir=repo_root)
    # audio_filename = cfg["dataset"].get("audio_filename", "song.mp3")
    # metadata_filename = cfg["dataset"].get("metadata_filename", "song.json")

    out_dir = resolve_path(cfg["preprocess"]["out_dir"], base_dir=repo_root)
    sample_rate = int(cfg["preprocess"].get("sample_rate", 16000))
    channels = int(cfg["preprocess"].get("channels", 1))

    seg_cfg = cfg["preprocess"].get("segment", {}) or {}
    segment_enabled = bool(seg_cfg.get("enabled", True))
    segment_seconds = float(seg_cfg.get("segment_seconds", 10.0))
    hop_seconds = float(seg_cfg.get("hop_seconds", segment_seconds))
    min_segment_seconds = float(seg_cfg.get("min_segment_seconds", min(1.0, segment_seconds)))

    ncfg_dict = cfg["preprocess"].get("normalize", {}) or {}
    ncfg = NormalizeConfig(
        method=str(ncfg_dict.get("method", "rms")),
        target_rms_db=float(ncfg_dict.get("target_rms_db", -20.0)),
        peak_db=float(ncfg_dict.get("peak_db", -1.0)),
        target_lufs=float(ncfg_dict.get("target_lufs", -18.0)),
    )

    manifest_path = resolve_path(cfg["manifest"]["out_path"], base_dir=repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # meta_paths = list(dataset_root.rglob(metadata_filename))
    meta_paths = list(dataset_root.rglob("*.json"))

    if args.max_songs is not None:
        meta_paths = meta_paths[: max(0, int(args.max_songs))]

    with manifest_path.open("w", encoding="utf-8") as mf:
        for meta_path in tqdm(meta_paths, desc="Preprocessing songs"):
            song_dir = meta_path.parent
            # audio_path = song_dir / audio_filename
            # if not audio_path.exists():
            #     continue

            # Find matching mp3 in same folder
            mp3_files = list(song_dir.glob("*.mp3"))

            if not mp3_files:
                continue

            audio_path = mp3_files[0]  # take first match


            try:
                meta = _read_json(meta_path)
            except Exception:
                continue

            mbid = str(meta.get("mbid", "")).strip()
            if not mbid:
                continue

            duration_sec = _get_duration_sec(meta, audio_path=audio_path)
            raga, tala, artist = extract_conditioning(meta)

            if not segment_enabled:
                starts = [0.0]
                seg_durs = [duration_sec]
            else:
                starts = []
                seg_durs = []
                t = 0.0
                while t < duration_sec:
                    d = min(segment_seconds, max(0.0, duration_sec - t))
                    if d >= min_segment_seconds:
                        starts.append(t)
                        seg_durs.append(d)
                    t += hop_seconds

            for seg_idx, (start, seg_dur) in enumerate(zip(starts, seg_durs, strict=True)):
                wav_path = out_dir / mbid / f"seg_{seg_idx:05d}.wav"
                if wav_path.exists() and wav_path.stat().st_size > 0 and not (args.overwrite):
                    pass
                else:
                    decode_segment_to_wav(
                        audio_path,
                        out_wav=wav_path,
                        start_sec=float(start),
                        duration_sec=float(seg_dur),
                        sample_rate=sample_rate,
                        channels=channels,
                    )

                    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
                    if sr != sample_rate:
                        raise RuntimeError(f"Expected sr={sample_rate}, got {sr} for {wav_path}")
                    audio = normalize_audio(audio, sample_rate=sr, cfg=ncfg)
                    sf.write(wav_path, audio, sr, subtype="PCM_16")

                row = {
                    "mbid": mbid,
                    "segment_index": seg_idx,
                    "start_sec": float(start),
                    "duration_sec": float(seg_dur),
                    "wav_path": _maybe_rel(wav_path, base=repo_root),
                    "metadata_path": _maybe_rel(meta_path, base=repo_root),
                    "raga": raga,
                    "tala": tala,
                    "artist": artist,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
