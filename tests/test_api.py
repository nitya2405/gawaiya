"""
tests/test_api.py

Smoke tests for the FastAPI backend.
These tests mock the Celery task and model so they run without GPU or Redis.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """
    Patch heavy dependencies before importing the app so tests run without
    Redis, GPU, or a real model checkpoint.
    """
    fake_redis = MagicMock()
    fake_redis.hgetall.return_value = {}
    fake_redis.lrange.return_value = []
    fake_redis.llen.return_value = 0

    with (
        patch("backend.worker._redis", fake_redis),
        patch("backend.main._redis", fake_redis),
        patch("backend.rate_limit.redis_lib.Redis", return_value=fake_redis),
    ):
        from backend.main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /api/ragas
# ---------------------------------------------------------------------------


def test_ragas_returns_list(client):
    resp = client.get("/api/ragas")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 60
    # Every entry has required keys
    for item in data:
        assert "name" in item
        assert "thaat" in item
        assert "time" in item
        assert "mood" in item


def test_ragas_contains_kalyan(client):
    resp = client.get("/api/ragas")
    names = [r["name"] for r in resp.json()]
    assert "Kalyāṇ" in names


# ---------------------------------------------------------------------------
# /api/talas
# ---------------------------------------------------------------------------


def test_talas_returns_list(client):
    resp = client.get("/api/talas")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 8
    for item in data:
        assert "name" in item
        assert "beats" in item


def test_talas_contains_tintal(client):
    resp = client.get("/api/talas")
    names = [t["name"] for t in resp.json()]
    assert "Tīntāl" in names


# ---------------------------------------------------------------------------
# /api/generate — validation
# ---------------------------------------------------------------------------


def test_generate_invalid_raga(client):
    resp = client.post("/api/generate", json={"raga": "NotARaga", "tala": "Tīntāl"})
    assert resp.status_code == 422


def test_generate_invalid_tala(client):
    resp = client.post("/api/generate", json={"raga": "Kalyāṇ", "tala": "NotATala"})
    assert resp.status_code == 422


def test_generate_duration_out_of_range(client):
    resp = client.post("/api/generate", json={"raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 120})
    assert resp.status_code == 422


def test_generate_invalid_codebooks(client):
    resp = client.post("/api/generate", json={"raga": "Kalyāṇ", "tala": "Tīntāl", "n_codebooks": 3})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/generate — success (mocked task)
# ---------------------------------------------------------------------------


def test_generate_enqueues_and_returns_job_id(client):
    job_id = str(uuid.uuid4())

    with (
        patch("backend.main.generate_task") as mock_task,
        patch("backend.main._redis") as mock_redis,
        patch("backend.main.check_rate_limit", return_value=(True, 9)),
    ):
        mock_redis.lrange.return_value = []
        mock_redis.hgetall.return_value = {}
        mock_task.apply_async.return_value = MagicMock(id=job_id)

        resp = client.post(
            "/api/generate",
            json={"raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 12},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert len(data["job_id"]) == 36   # UUID length


# ---------------------------------------------------------------------------
# /api/job/{id} — not found
# ---------------------------------------------------------------------------


def test_job_not_found(client):
    with patch("backend.main._read_job", return_value=None):
        resp = client.get(f"/api/job/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/job/{id} — queued state
# ---------------------------------------------------------------------------


def test_job_queued_state(client):
    job_id = str(uuid.uuid4())
    with (
        patch("backend.main._read_job", return_value={
            "status": "queued", "progress": 0.0, "error": None
        }),
        patch("backend.main._queue_position", return_value=2),
    ):
        resp = client.get(f"/api/job/{job_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert data["queue_position"] == 2


# ---------------------------------------------------------------------------
# /api/feedback/{id}
# ---------------------------------------------------------------------------


def test_feedback_thumbs_up(client):
    job_id = str(uuid.uuid4())
    with (
        patch("backend.main._read_job", return_value={
            "status": "done", "raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 12
        }),
        patch("backend.main._redis") as mock_redis,
    ):
        resp = client.get(f"/api/feedback/{job_id}?value=1")

    assert resp.status_code == 204


def test_feedback_invalid_value(client):
    job_id = str(uuid.uuid4())
    with patch("backend.main._read_job", return_value={"status": "done"}):
        resp = client.get(f"/api/feedback/{job_id}?value=0")
    assert resp.status_code == 422
