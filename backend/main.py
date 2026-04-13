"""
backend/main.py

FastAPI app — self-contained, no Celery, no Redis.
Generation runs in a background thread. Job state lives in memory.

Run with:
    python -m uvicorn backend.main:app --reload
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import (
    CLIP_SEC,
    CROSSFADE_SEC,
    MP3_BITRATE,
    OUTPUTS_DIR,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
    REPO_ROOT,
)
from backend.raga_meta import get_raga_list, get_tala_list, raga_names, tala_names
from backend.schemas import GenerateRequest, GenerateResponse, JobStatus

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Sangeet AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory rate limiter (sliding window, no Redis required)
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, remaining). Sliding window over RATE_LIMIT_WINDOW seconds."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    with _rate_lock:
        timestamps = [t for t in _rate_store.get(ip, []) if t > cutoff]
        timestamps.append(now)
        _rate_store[ip] = timestamps
    count = len(timestamps)
    remaining = max(0, RATE_LIMIT_MAX - count)
    return count <= RATE_LIMIT_MAX, remaining


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return getattr(request.client, "host", "0.0.0.0")


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_queue: deque[str] = deque()
_lock = threading.Lock()

# Single background worker thread
_worker_thread: Optional[threading.Thread] = None
_worker_running = False


def _get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _queue_position(job_id: str) -> int | None:
    with _lock:
        q = list(_queue)
    try:
        return q.index(job_id) + 1
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Crossfade helper
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
# Generation (runs inside worker thread)
# ---------------------------------------------------------------------------

def _run_generation(job_id: str) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from sangeet.audio.postprocess import postprocess_wav
    from sangeet.data.dataset import token_ids_to_codes
    from sangeet.tokenizer.encodec_codec import decode_codes_to_wav
    from backend.model_cache import get_model

    job = _get_job(job_id)
    raga        = job["raga"]
    tala        = job["tala"]
    duration_sec = job["duration_sec"]
    cfg_scale   = job["cfg_scale"]
    n_codebooks = job["n_codebooks"]

    _update_job(job_id, status="running", progress=0.0, queue_position=None)

    try:
        import torch
        cache        = get_model()
        model        = cache["model"]
        token_meta   = cache["token_meta"]
        raga_vocab   = cache["raga_vocab"]
        tala_vocab   = cache["tala_vocab"]
        artist_vocab = cache["artist_vocab"]
        enc_model    = cache["enc_model"]
        device       = cache["device"]

        def _safe_encode(vocab, key):
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

        n_clips = max(1, math.ceil(duration_sec / CLIP_SEC))
        clips: list[np.ndarray] = []
        sr_out = None

        _update_job(job_id, n_clips=n_clips, clip_num=0)

        for i in range(n_clips):
            # Check for cancellation between clips
            if _get_job(job_id).get("status") == "cancelled":
                return

            _update_job(job_id, clip_num=i + 1)
            n_frames  = max(1, int(CLIP_SEC * frame_rate))
            cb_scales = CB_TEMPERATURE_SCALES[:n_cb_full]

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
            _update_job(job_id, progress=(i + 1) / n_clips)

        # Stitch
        combined = clips[0]
        for clip in clips[1:]:
            combined = _crossfade(combined, clip, sr_out, CROSSFADE_SEC)
        combined = combined[:int(duration_sec * sr_out)]

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = OUTPUTS_DIR / f"{job_id}.wav"
        sf.write(str(wav_path), combined, sr_out, subtype="PCM_16")

        postprocess_wav(wav_path, wav_path)

        mp3_path = OUTPUTS_DIR / f"{job_id}.mp3"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, str(mp3_path)],
            capture_output=True,
        )
        wav_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg: {result.stderr.decode()}")

        _update_job(job_id, status="done", progress=1.0, output_path=str(mp3_path))

    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------------
# Worker loop (single thread, processes queue in order)
# ---------------------------------------------------------------------------

def _worker_loop() -> None:
    global _worker_running
    _worker_running = True
    while True:
        with _lock:
            job_id = _queue.popleft() if _queue else None
        if job_id is None:
            threading.Event().wait(0.5)   # sleep 0.5s, check again
            continue
        _run_generation(job_id)


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(application):
    global _worker_thread
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="generation-worker")
    _worker_thread.start()
    yield

app.router.lifespan_context = _lifespan


# ---------------------------------------------------------------------------
# Vocab endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ragas")
def list_ragas():
    return get_raga_list()


@app.get("/api/talas")
def list_talas():
    return get_tala_list()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@app.post("/api/generate", response_model=GenerateResponse, status_code=202)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    if req.raga not in raga_names():
        raise HTTPException(status_code=422, detail=f"Unknown raga: '{req.raga}'. See /api/ragas.")
    if req.tala not in tala_names():
        raise HTTPException(status_code=422, detail=f"Unknown tala: '{req.tala}'. See /api/talas.")

    ip = _get_client_ip(request)
    allowed, _ = _check_rate_limit(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {RATE_LIMIT_MAX} generations per hour.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status":       "queued",
            "progress":     0.0,
            "error":        None,
            "output_path":  None,
            "raga":         req.raga,
            "tala":         req.tala,
            "duration_sec": req.duration_sec,
            "cfg_scale":    req.cfg_scale,
            "n_codebooks":  req.n_codebooks,
            "n_clips":      0,
            "clip_num":     0,
        }
        _queue.append(job_id)

    return GenerateResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get("/api/job/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    queue_pos = _queue_position(job_id) if job["status"] == "queued" else None

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=float(job["progress"]),
        queue_position=queue_pos,
        error=job.get("error"),
        n_clips=int(job.get("n_clips", 0)),
        clip_num=int(job.get("clip_num", 0)),
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

_AUDIO_FORMATS = {
    "mp3":  ("audio/mpeg",  ["-codec:a", "libmp3lame", "-b:a", "192k"]),
    "wav":  ("audio/wav",   ["-codec:a", "pcm_s16le"]),
    "flac": ("audio/flac",  ["-codec:a", "flac"]),
    "ogg":  ("audio/ogg",   ["-codec:a", "libvorbis", "-q:a", "6"]),
}


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str, format: str = "mp3") -> StreamingResponse:
    fmt = format.lower()
    if fmt not in _AUDIO_FORMATS:
        raise HTTPException(status_code=422, detail=f"Unsupported format '{fmt}'. Choose from: {', '.join(_AUDIO_FORMATS)}")

    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet.")

    mp3_path = Path(job["output_path"])
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing.")

    media_type, codec_args = _AUDIO_FORMATS[fmt]

    if fmt == "mp3":
        # Serve the stored file directly — no re-encode needed
        def _iter():
            with open(mp3_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return StreamingResponse(
            _iter(),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="sangeet-{job_id}.mp3"'},
        )

    # Transcode from stored MP3 to requested format via ffmpeg pipe
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        *codec_args,
        "-f", fmt if fmt != "ogg" else "ogg",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _iter_proc():
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.stdout.close()
            proc.wait()

    return StreamingResponse(
        _iter_proc(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="sangeet-{job_id}.{fmt}"'},
    )


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


@app.post("/api/cancel/{job_id}", status_code=204)
def cancel_job(job_id: str) -> Response:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] in ("queued", "running"):
        _update_job(job_id, status="cancelled")
        with _lock:
            try:
                _queue.remove(job_id)
            except ValueError:
                pass
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Share link
# ---------------------------------------------------------------------------


@app.get("/api/share/{job_id}")
def get_share(job_id: str) -> dict:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return {
        "job_id":       job_id,
        "raga":         job.get("raga"),
        "tala":         job.get("tala"),
        "duration_sec": job.get("duration_sec"),
        "status":       job.get("status"),
        "audio_url":    f"/api/audio/{job_id}" if job.get("status") == "done" else None,
    }


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.get("/api/feedback/{job_id}")
def submit_feedback(job_id: str, value: int) -> Response:
    if value not in (1, -1):
        raise HTTPException(status_code=422, detail="value must be 1 or -1")
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    import datetime, json
    record = {
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "job_id":       job_id,
        "raga":         job.get("raga"),
        "tala":         job.get("tala"),
        "duration_sec": job.get("duration_sec"),
        "rating":       value,
    }
    from backend.config import FEEDBACK_LOG
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/api/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            job = _get_job(job_id)
            if job is None:
                await websocket.send_json({"error": "job not found"})
                break

            queue_pos = _queue_position(job_id) if job["status"] == "queued" else None
            await websocket.send_json({
                "status":         job["status"],
                "progress":       float(job["progress"]),
                "queue_position": queue_pos,
                "error":          job.get("error"),
                "n_clips":        int(job.get("n_clips", 0)),
                "clip_num":       int(job.get("clip_num", 0)),
            })

            if job["status"] in ("done", "failed"):
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
