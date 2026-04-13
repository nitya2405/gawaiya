"""
backend/config.py

All environment variables and path configuration.
Override any of these with environment variables at runtime.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Model checkpoint — swap this single line to deploy Phase 1.5 delay-pattern model
MODEL_CHECKPOINT = Path(
    os.environ.get("MODEL_CHECKPOINT", str(REPO_ROOT / "runs/hindustani_cfg/checkpoints/latest.pt"))
)

VOCABS_DIR = Path(
    os.environ.get("VOCABS_DIR", str(REPO_ROOT / "runs/hindustani_small/vocabs"))
)

OUTPUTS_DIR = Path(
    os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "outputs/api"))
)

FEEDBACK_LOG = Path(
    os.environ.get("FEEDBACK_LOG", str(REPO_ROOT / "feedback.jsonl"))
)

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------

CELERY_BROKER_URL   = os.environ.get("CELERY_BROKER_URL",   REDIS_URL)
CELERY_BACKEND_URL  = os.environ.get("CELERY_BACKEND_URL",  REDIS_URL)

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

DEFAULT_DURATION_SEC = 12
DEFAULT_CFG_SCALE    = 5.0
DEFAULT_N_CODEBOOKS  = 4
CLIP_SEC             = 12.0          # context-safe clip length
CROSSFADE_SEC        = 1.0

MIN_DURATION_SEC     = 6
MAX_DURATION_SEC     = 60
MIN_CFG_SCALE        = 3.0
MAX_CFG_SCALE        = 7.0
VALID_N_CODEBOOKS    = {2, 4, 8}

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_MAX       = int(os.environ.get("RATE_LIMIT_MAX",   "10"))   # requests
RATE_LIMIT_WINDOW    = int(os.environ.get("RATE_LIMIT_WINDOW", "3600")) # seconds (1 hour)

# ---------------------------------------------------------------------------
# Job TTL
# ---------------------------------------------------------------------------

JOB_TTL_SECONDS = 3600   # 1 hour

# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

MP3_BITRATE = "192k"
