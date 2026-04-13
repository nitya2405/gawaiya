"""
backend/rate_limit.py

Redis token bucket rate limiter.
10 generations per IP per hour (not a hard block — CGNAT-safe sliding window).
"""

from __future__ import annotations

import time
from typing import Optional

import redis as redis_lib

from backend.config import RATE_LIMIT_MAX, RATE_LIMIT_WINDOW


def _bucket_key(ip: str) -> str:
    return f"rate:{ip}"


def check_rate_limit(redis: redis_lib.Redis, ip: str) -> tuple[bool, int]:
    """
    Returns (allowed: bool, remaining: int).
    Uses a sliding window with a sorted set (timestamp as score).
    CGNAT-safe: we soft-limit, not hard block (caller decides what to do).
    """
    key    = _bucket_key(ip)
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window)          # drop expired entries
    pipe.zadd(key, {str(now): now})                     # add current request
    pipe.zcard(key)                                     # count in window
    pipe.expire(key, RATE_LIMIT_WINDOW)
    results = pipe.execute()

    count: int = results[2]
    remaining  = max(0, RATE_LIMIT_MAX - count)
    allowed    = count <= RATE_LIMIT_MAX
    return allowed, remaining


def get_client_ip(request) -> str:
    """Extract IP from request, respecting X-Forwarded-For (Cloudflare/proxy)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host or "0.0.0.0"
