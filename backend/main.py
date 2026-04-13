"""
backend/main.py

FastAPI application — all HTTP + WebSocket routes.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import JOB_TTL_SECONDS, OUTPUTS_DIR, REDIS_URL
from backend.rate_limit import check_rate_limit, get_client_ip
from backend.raga_meta import get_raga_list, get_tala_list, raga_names, tala_names
from backend.schemas import GenerateRequest, GenerateResponse, JobStatus
from backend.worker import QUEUE_KEY, _job_key, _redis, _set_job, celery_app, generate_task

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Sangeet AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_job(job_id: str) -> dict | None:
    key  = _job_key(job_id)
    data = _redis.hgetall(key)
    if not data:
        return None
    return {k: json.loads(v) for k, v in data.items()}


def _queue_position(job_id: str) -> int | None:
    """1-based position in queue, or None if not in queue."""
    members = _redis.lrange(QUEUE_KEY, 0, -1)
    try:
        return members.index(job_id) + 1
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Vocab endpoints
# ---------------------------------------------------------------------------


@app.get("/api/ragas")
def list_ragas() -> list[dict]:
    return get_raga_list()


@app.get("/api/talas")
def list_talas() -> list[dict]:
    return get_tala_list()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=GenerateResponse, status_code=202)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    # Validate raga/tala before consuming a rate-limit token
    if req.raga not in raga_names():
        raise HTTPException(status_code=422, detail=f"Unknown raga: '{req.raga}'. See /api/ragas.")
    if req.tala not in tala_names():
        raise HTTPException(status_code=422, detail=f"Unknown tala: '{req.tala}'. See /api/talas.")

    ip = get_client_ip(request)
    allowed, remaining = check_rate_limit(_redis, ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {10} generations per hour per IP.",
            headers={"Retry-After": "3600"},
        )

    job_id = str(uuid.uuid4())

    # Write initial job state to Redis
    _set_job(job_id, {
        "status":         "queued",
        "progress":       0.0,
        "queue_position": None,
        "error":          None,
        "output_path":    None,
        "raga":           req.raga,
        "tala":           req.tala,
        "duration_sec":   req.duration_sec,
        "cfg_scale":      req.cfg_scale,
        "n_codebooks":    req.n_codebooks,
    })

    # Push to queue list for position tracking
    _redis.rpush(QUEUE_KEY, job_id)

    # Enqueue Celery task
    generate_task.apply_async(
        args=[job_id, req.raga, req.tala, req.duration_sec, req.cfg_scale, req.n_codebooks],
        task_id=job_id,
    )

    return GenerateResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@app.get("/api/job/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    data = _read_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")

    queue_pos = _queue_position(job_id) if data.get("status") == "queued" else None

    return JobStatus(
        job_id=job_id,
        status=data.get("status", "queued"),
        progress=float(data.get("progress", 0.0)),
        queue_position=queue_pos,
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------


def _iter_file(path: Path, chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:
    async def _gen():
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
    return _gen()


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str) -> StreamingResponse:
    data = _read_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    if data.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet.")

    mp3_path = Path(data["output_path"])
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing.")

    return StreamingResponse(
        _iter_file(mp3_path),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.mp3"',
            "Cache-Control": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@app.get("/api/feedback/{job_id}")
def submit_feedback(job_id: str, value: int, request: Request) -> Response:
    if value not in (1, -1):
        raise HTTPException(status_code=422, detail="value must be 1 (👍) or -1 (👎)")

    data = _read_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")

    import datetime, json as _json
    record = {
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "job_id":       job_id,
        "raga":         data.get("raga"),
        "tala":         data.get("tala"),
        "duration_sec": data.get("duration_sec"),
        "rating":       value,
    }

    # Buffer to Redis list; background thread (or separate process) flushes to disk
    _redis.rpush("feedback_queue", _json.dumps(record))

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# WebSocket progress stream
# ---------------------------------------------------------------------------


@app.websocket("/api/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            data = _read_job(job_id)
            if data is None:
                await websocket.send_json({"error": "job not found"})
                break

            queue_pos = _queue_position(job_id) if data.get("status") == "queued" else None

            payload = {
                "status":         data.get("status"),
                "progress":       float(data.get("progress", 0.0)),
                "queue_position": queue_pos,
                "error":          data.get("error"),
            }
            await websocket.send_json(payload)

            if data.get("status") in ("done", "failed"):
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Feedback flush background task
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager
import threading


def _flush_feedback_loop(stop_event: threading.Event) -> None:
    """Drains Redis feedback_queue → feedback.jsonl every 60s."""
    from backend.config import FEEDBACK_LOG
    import time

    while not stop_event.is_set():
        time.sleep(60)
        records = []
        while True:
            raw = _redis.lpop("feedback_queue")
            if raw is None:
                break
            records.append(raw)
        if records:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(r + "\n")


_stop_event = threading.Event()
_flush_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _flush_thread
    _flush_thread = threading.Thread(target=_flush_feedback_loop, args=(_stop_event,), daemon=True)
    _flush_thread.start()
    yield
    _stop_event.set()


app.router.lifespan_context = lifespan
