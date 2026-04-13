from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """
    Best-effort repo root detection for running scripts from subdirectories.

    Heuristic: walk up from `start` until a marker file is found.
    """
    markers = {"requirements.txt", "pyproject.toml", ".git"}
    cur = Path(start) if start is not None else Path.cwd()
    cur = cur.resolve()

    for parent in [cur, *cur.parents]:
        for m in markers:
            if (parent / m).exists():
                return parent

    return cur

