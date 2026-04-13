from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from sangeet.data.vocab import Vocab
from sangeet.utils.jsonl import read_jsonl


@dataclass(frozen=True)
class TokenSpec:
    n_codebooks: int
    codebook_size: int
    token_offset: int = 2  # 0=PAD, 1=BOS
    pad_id: int = 0
    bos_id: int = 1

    @property
    def vocab_size(self) -> int:
        return int(self.token_offset + self.n_codebooks * self.codebook_size)


def codes_to_token_ids(codes: np.ndarray, spec: TokenSpec) -> np.ndarray:
    """
    Convert Encodec codes [K, T] -> flattened token ids [T*K].
    """
    if codes.ndim != 2:
        raise ValueError(f"Expected codes shape [K,T], got {codes.shape}")
    k, t = int(codes.shape[0]), int(codes.shape[1])
    if k != spec.n_codebooks:
        raise ValueError(f"n_codebooks mismatch: codes={k}, spec={spec.n_codebooks}")

    frame_codes = codes.T.astype(np.int64)  # [T, K]
    offsets = (np.arange(k, dtype=np.int64) * int(spec.codebook_size))[None, :]  # [1, K]
    flat = (frame_codes + offsets).reshape(t * k)
    return flat + int(spec.token_offset)


def token_ids_to_codes(token_ids: np.ndarray, spec: TokenSpec) -> np.ndarray:
    """
    Convert flattened token ids [T*K] -> codes [K, T].
    """
    tok = token_ids.astype(np.int64) - int(spec.token_offset)
    k = int(spec.n_codebooks)
    if tok.size % k != 0:
        raise ValueError(f"Token length must be divisible by n_codebooks={k}, got {tok.size}")
    t = tok.size // k
    tok = tok.reshape(t, k)  # [T, K]
    offsets = (np.arange(k, dtype=np.int64) * int(spec.codebook_size))[None, :]
    frame_codes = tok - offsets
    return frame_codes.T.astype(np.int16)


class CarnaticTokenDataset(Dataset):
    """
    Dataset over Encodec token files + conditioning.

    Expects a JSONL manifest with at least:
      - tokens_path (relative or absolute)
      - mbid
      - raga, tala, artist (optional; otherwise read from metadata_path)
      - metadata_path (optional)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repo_root: str | Path,
        token_spec: TokenSpec,
        raga_vocab: Vocab,
        tala_vocab: Vocab,
        artist_vocab: Vocab,
        max_seq_len: int | None = None,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repo_root = Path(repo_root)
        self.token_spec = token_spec
        self.raga_vocab = raga_vocab
        self.tala_vocab = tala_vocab
        self.artist_vocab = artist_vocab
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        self.rng = np.random.default_rng(int(seed))

        self.rows = list(read_jsonl(self.manifest_path))
        if not self.rows:
            raise FileNotFoundError(f"Empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_tokens_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        # token paths are usually relative to repo root.
        cand = (self.repo_root / path).resolve()
        if cand.exists():
            return cand
        # or relative to the manifest directory.
        return (self.manifest_path.parent / path).resolve()

    def _maybe_read_metadata(self, row: dict[str, Any]) -> dict[str, Any] | None:
        meta_path = row.get("metadata_path")
        if not meta_path:
            return None
        p = Path(meta_path)
        if not p.is_absolute():
            p = (self.repo_root / p).resolve()
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[int(idx)]
        tok_path = self._resolve_tokens_path(str(row["tokens_path"]))

        with np.load(tok_path, allow_pickle=False) as z:
            codes = z["codes"]  # [K, T]

        token_ids = codes_to_token_ids(codes, self.token_spec).astype(np.int64)

        k = self.token_spec.n_codebooks
        if token_ids.size % k != 0:
            raise RuntimeError(f"Token length not divisible by n_codebooks: {tok_path}")

        if self.max_seq_len is not None:
            # Keep sequences aligned to complete Encodec frames.
            max_frames = max(1, (self.max_seq_len // k))
            n_frames = token_ids.size // k
            if n_frames > max_frames:
                start_frame = int(self.rng.integers(0, n_frames - max_frames + 1))
                start = start_frame * k
                end = start + max_frames * k
                token_ids = token_ids[start:end]

        raga = row.get("raga")
        tala = row.get("tala")
        artist = row.get("artist")
        if (raga is None) or (tala is None) or (artist is None):
            meta = self._maybe_read_metadata(row) or {}
            raga = raga or meta.get("raga") or meta.get("raaga") or "unknown"
            tala = tala or meta.get("tala") or meta.get("taala") or "unknown"
            artist = artist or "unknown"

        return {
            "token_ids": torch.from_numpy(token_ids).long(),
            "raga_id": torch.tensor(self.raga_vocab.encode(str(raga)), dtype=torch.long),
            "tala_id": torch.tensor(self.tala_vocab.encode(str(tala)), dtype=torch.long),
            "artist_id": torch.tensor(self.artist_vocab.encode(str(artist)), dtype=torch.long),
        }


def collate_lm(batch: list[dict[str, torch.Tensor]], *, token_spec: TokenSpec) -> dict[str, torch.Tensor]:
    pad_id = int(token_spec.pad_id)
    bos_id = int(token_spec.bos_id)

    tokens = [b["token_ids"] for b in batch]
    lengths = torch.tensor([t.numel() for t in tokens], dtype=torch.long)
    max_len = int(lengths.max().item())

    token_mat = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        token_mat[i, : t.numel()] = t

    # Teacher forcing inputs: [BOS] + tokens[:-1]
    input_ids = torch.full((len(batch), max_len), bos_id, dtype=torch.long)
    input_ids[:, 1:] = token_mat[:, :-1]

    target_ids = token_mat

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "lengths": lengths,
        "raga_id": torch.stack([b["raga_id"] for b in batch], dim=0),
        "tala_id": torch.stack([b["tala_id"] for b in batch], dim=0),
        "artist_id": torch.stack([b["artist_id"] for b in batch], dim=0),
    }

