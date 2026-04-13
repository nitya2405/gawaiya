"""
backend/worker.py

Celery worker — runs on the GPU machine.
One task at a time (GPU constraint): concurrency=1.

Start with:
    celery -A backend.worker worker --loglevel=info --concurrency=1
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import redis as redis_lib
import soundfile as sf

from celery import Celery

from backend.config import (
    CELERY_BACKEND_URL,
    CELERY_BROKER_URL,
    CLIP_SEC,
    CROSSFADE_SEC,
    JOB_TTL_SECONDS,
    MP3_BITRATE,
    OUTPUTS_DIR,
    REDIS_URL,
    REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "sangeet",
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

_redis = redis_lib.from_url(REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _set_job(job_id: str, data: dict, ttl: int = JOB_TTL_SECONDS) -> None:
    key = _job_key(job_id)
    _redis.hset(key, mapping={k: json.dumps(v) for k, v in data.items()})
    _redis.expire(key, ttl)


def _update_job(job_id: str, **kwargs) -> None:
    key = _job_key(job_id)
    _redis.hset(key, mapping={k: json.dumps(v) for k, v in kwargs.items()})
    _redis.expire(key, JOB_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Queue position helpers (maintained by main.py on enqueue)
# ---------------------------------------------------------------------------

QUEUE_KEY = "job_queue"


def queue_length() -> int:
    return _redis.llen(QUEUE_KEY)


# ---------------------------------------------------------------------------
# Crossfade util (mirrors generate_music.py)
# ---------------------------------------------------------------------------

def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_sec: float) -> np.ndarray:
    n = min(int(fade_sec * sr), len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b], axis=0)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if a.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in  = fade_in[:, None]
    return np.concatenate([a[:-n], a[-n:] * fade_out + b[:n] * fade_in, b[n:]], axis=0)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="sangeet.generate")
def generate_task(
    self,
    job_id: str,
    raga: str,
    tala: str,
    duration_sec: int,
    cfg_scale: float,
    n_codebooks: int,
) -> None:
    """
    Full generation pipeline:
      1. Load model (cached in worker process memory after first call)
      2. Generate clips, stitch with crossfades
      3. Post-process (HF rolloff + LUFS normalization)
      4. Transcode WAV → MP3 via ffmpeg
      5. Write final path to Redis job record
    """
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from sangeet.audio.postprocess import postprocess_wav
    from sangeet.data.dataset import token_ids_to_codes
    from sangeet.tokenizer.encodec_codec import decode_codes_to_wav
    from backend.model_cache import get_model

    _update_job(job_id, status="running", progress=0.0, queue_position=None)
    _redis.lrem(QUEUE_KEY, 0, job_id)   # remove from queue list

    try:
        cache      = get_model()
        model      = cache["model"]
        token_meta = cache["token_meta"]
        raga_vocab = cache["raga_vocab"]
        tala_vocab = cache["tala_vocab"]
        artist_vocab = cache["artist_vocab"]
        enc_model  = cache["enc_model"]
        device     = cache["device"]

        # Encode conditioning
        def _safe_encode(vocab, key: str) -> int:
            if key in vocab.stoi:
                return vocab.encode(key)
            for k in vocab.stoi:
                if k.lower() == key.lower():
                    return vocab.encode(k)
            return vocab.encode("unknown")

        raga_id   = _safe_encode(raga_vocab,   raga)
        tala_id   = _safe_encode(tala_vocab,   tala)
        artist_id = _safe_encode(artist_vocab, "unknown")

        frame_rate  = float(token_meta.get("frame_rate", 75.0))
        n_cb_full   = int(token_meta["n_codebooks"])
        sample_rate = int(token_meta["encodec_sample_rate"])
        token_spec  = model.token_spec

        CB_TEMPERATURE_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        n_clips   = max(1, math.ceil(duration_sec / CLIP_SEC))
        clips: list[np.ndarray] = []
        sr_out    = None

        for i in range(n_clips):
            n_frames = max(1, int(CLIP_SEC * frame_rate))
            cb_scales = CB_TEMPERATURE_SCALES[:n_cb_full]

            import torch
            with torch.inference_mode():
                token_ids = model.generate(
                    raga_id=raga_id,
                    tala_id=tala_id,
                    artist_id=artist_id,
                    n_frames=n_frames,
                    temperature=0.75,
                    top_p=0.9,
                    cfg_scale=cfg_scale,
                    cb_temperature_scales=cb_scales,
                    device=device,
                )

            codes = token_ids_to_codes(
                token_ids.detach().cpu().numpy().astype(np.int64),
                token_spec,
            )
            if n_codebooks < n_cb_full:
                codes[n_codebooks:, :] = 0

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)
            try:
                decode_codes_to_wav(enc_model, codes=codes, out_wav_path=tmp_path, sample_rate=sample_rate)
                audio, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
            finally:
                os.unlink(tmp_path)

            clips.append(audio)
            sr_out = sr
            progress = (i + 1) / n_clips
            _update_job(job_id, progress=progress)

        # Stitch
        combined = clips[0]
        for clip in clips[1:]:
            combined = _crossfade(combined, clip, sr_out, CROSSFADE_SEC)
        combined = combined[:int(duration_sec * sr_out)]

        # Write WAV
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = OUTPUTS_DIR / f"{job_id}.wav"
        sf.write(str(wav_path), combined, sr_out, subtype="PCM_16")

        # Post-process in-place
        postprocess_wav(wav_path, wav_path)

        # Transcode WAV → MP3
        mp3_path = OUTPUTS_DIR / f"{job_id}.mp3"
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", MP3_BITRATE,
                str(mp3_path),
            ],
            capture_output=True,
        )
        wav_path.unlink(missing_ok=True)   # delete WAV after successful transcode

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

        _update_job(job_id, status="done", progress=1.0, output_path=str(mp3_path))

    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))
        raise
