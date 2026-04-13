"""
backend/schemas.py

Pydantic request/response models for the API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.config import (
    DEFAULT_CFG_SCALE,
    DEFAULT_DURATION_SEC,
    DEFAULT_N_CODEBOOKS,
    MAX_CFG_SCALE,
    MAX_DURATION_SEC,
    MIN_CFG_SCALE,
    MIN_DURATION_SEC,
    VALID_N_CODEBOOKS,
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    raga: str = Field("Kalyāṇ", description="Raga name (from /api/ragas)")
    tala: str = Field("Tīntāl", description="Tala name (from /api/talas)")
    duration_sec: int = Field(
        DEFAULT_DURATION_SEC,
        ge=MIN_DURATION_SEC,
        le=MAX_DURATION_SEC,
        description="Output duration in seconds (6–60)",
    )
    cfg_scale: float = Field(
        DEFAULT_CFG_SCALE,
        ge=MIN_CFG_SCALE,
        le=MAX_CFG_SCALE,
        description="Classifier-free guidance scale (3.0–7.0)",
    )
    n_codebooks: int = Field(
        DEFAULT_N_CODEBOOKS,
        description="Number of codebooks to generate (2, 4, or 8)",
    )

    @field_validator("n_codebooks")
    @classmethod
    def validate_codebooks(cls, v: int) -> int:
        if v not in VALID_N_CODEBOOKS:
            raise ValueError(f"n_codebooks must be one of {sorted(VALID_N_CODEBOOKS)}")
        return v


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class GenerateResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    progress: float = Field(0.0, ge=0.0, le=1.0, description="0.0–1.0")
    queue_position: Optional[int] = Field(None, description="Position in queue (1-based), None if running/done")
    error: Optional[str] = None


class RagaMeta(BaseModel):
    name: str
    thaat: str
    time: str        # e.g. "Evening", "Morning", "Any"
    mood: str        # e.g. "Serene", "Devotional", "Romantic"


class TalaMeta(BaseModel):
    name: str
    beats: int
    character: str   # e.g. "Medium", "Fast", "Slow"
