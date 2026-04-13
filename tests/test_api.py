"""
tests/test_api.py

Smoke tests for the FastAPI backend (no-Redis, thread-based generation).

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/ragas
# ---------------------------------------------------------------------------

def test_ragas_returns_list(client):
    resp = client.get("/api/ragas")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 60
    for item in data:
        assert {"name", "thaat", "time", "mood"} <= item.keys()


def test_ragas_contains_kalyan(client):
    names = [r["name"] for r in client.get("/api/ragas").json()]
    assert "Kalyāṇ" in names


# ---------------------------------------------------------------------------
# /api/talas
# ---------------------------------------------------------------------------

def test_talas_returns_list(client):
    resp = client.get("/api/talas")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 8
    for item in data:
        assert {"name", "beats"} <= item.keys()


def test_talas_contains_tintal(client):
    names = [t["name"] for t in client.get("/api/talas").json()]
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
# /api/generate — success path (no real generation)
# ---------------------------------------------------------------------------

def test_generate_returns_job_id(client):
    with patch("backend.main._check_rate_limit", return_value=(True, 9)):
        resp = client.post(
            "/api/generate",
            json={"raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 12},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert len(data["job_id"]) == 36  # UUID


def test_generate_rate_limited(client):
    with patch("backend.main._check_rate_limit", return_value=(False, 0)):
        resp = client.post(
            "/api/generate",
            json={"raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 12},
        )
    assert resp.status_code == 429


# ---------------------------------------------------------------------------
# /api/job/{id}
# ---------------------------------------------------------------------------

def test_job_not_found(client):
    resp = client.get(f"/api/job/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_job_queued_state(client):
    with patch("backend.main._check_rate_limit", return_value=(True, 9)):
        post = client.post("/api/generate", json={"raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 12})
    job_id = post.json()["job_id"]
    resp = client.get(f"/api/job/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("queued", "running", "done", "failed")
    assert "progress" in data


# ---------------------------------------------------------------------------
# /api/share/{id}
# ---------------------------------------------------------------------------

def test_share_not_found(client):
    resp = client.get(f"/api/share/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/feedback/{id}
# ---------------------------------------------------------------------------

def test_feedback_invalid_value(client):
    resp = client.get(f"/api/feedback/{uuid.uuid4()}?value=0")
    assert resp.status_code == 422


def test_feedback_job_not_found(client):
    resp = client.get(f"/api/feedback/{uuid.uuid4()}?value=1")
    assert resp.status_code == 404
