from __future__ import annotations

import hashlib
import re
from pathlib import Path


_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\\\|?*]+')


def safe_name(name: str, *, max_len: int = 80) -> str:
    """
    Create a Windows-safe directory/file name.

    Keeps it readable while preventing path issues.
    """
    if not name:
        return "unknown"

    name = name.strip().replace("\u0000", "")
    name = _WINDOWS_FORBIDDEN.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")

    if not name:
        return "unknown"

    if len(name) <= max_len:
        return name

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    trimmed = name[: max(1, max_len - 9)].rstrip(". ")
    return f"{trimmed}-{digest}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

