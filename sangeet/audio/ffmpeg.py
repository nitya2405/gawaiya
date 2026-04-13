from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFMpegNotFoundError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFMpegNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure `ffmpeg`/`ffprobe` are available."
        )
    if shutil.which("ffprobe") is None:
        raise FFMpegNotFoundError(
            "ffprobe not found on PATH. Install ffmpeg and ensure `ffprobe` is available."
        )


def ffprobe_duration_sec(path: str | Path) -> float:
    ensure_ffmpeg()
    path = Path(path)
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def decode_segment_to_wav(
    in_audio: str | Path,
    *,
    out_wav: str | Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int,
    channels: int,
) -> None:
    """
    Decode a slice of an audio file to PCM WAV using ffmpeg.
    """
    ensure_ffmpeg()
    in_audio = Path(in_audio)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(float(start_sec)),
        "-t",
        str(float(duration_sec)),
        "-i",
        str(in_audio),
        "-ac",
        str(int(channels)),
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {in_audio}: {proc.stderr.strip()}")

