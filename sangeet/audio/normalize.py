from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizeConfig:
    method: str = "rms"  # none|peak|rms|lufs
    target_rms_db: float = -20.0
    peak_db: float = -1.0
    target_lufs: float = -18.0


def _db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))


def peak_normalize(audio: np.ndarray, *, peak_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio
    target_peak = _db_to_amp(peak_db)
    return audio * (target_peak / peak)


def rms_normalize(audio: np.ndarray, *, target_rms_db: float, peak_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 0.0:
        return audio
    target_rms = _db_to_amp(target_rms_db)
    audio = audio * (target_rms / rms)
    return np.clip(audio, -_db_to_amp(peak_db), _db_to_amp(peak_db))


def lufs_normalize(audio: np.ndarray, *, sample_rate: int, target_lufs: float, peak_db: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln
    except Exception as e:
        raise RuntimeError("pyloudnorm is required for LUFS normalization. `pip install pyloudnorm`.") from e

    meter = pyln.Meter(sample_rate)
    loudness = float(meter.integrated_loudness(audio.astype(np.float64)))
    gain_db = float(target_lufs - loudness)
    audio = audio * _db_to_amp(gain_db)
    return np.clip(audio, -_db_to_amp(peak_db), _db_to_amp(peak_db))


def normalize_audio(audio: np.ndarray, *, sample_rate: int, cfg: NormalizeConfig) -> np.ndarray:
    method = (cfg.method or "none").lower()
    if method == "none":
        return audio
    if method == "peak":
        return peak_normalize(audio, peak_db=cfg.peak_db)
    if method == "rms":
        return rms_normalize(audio, target_rms_db=cfg.target_rms_db, peak_db=cfg.peak_db)
    if method == "lufs":
        return lufs_normalize(
            audio, sample_rate=sample_rate, target_lufs=cfg.target_lufs, peak_db=cfg.peak_db
        )
    raise ValueError(f"Unknown normalize method: {cfg.method}")

