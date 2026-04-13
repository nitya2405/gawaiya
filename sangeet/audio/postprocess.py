"""
sangeet/audio/postprocess.py

Post-processing pipeline for Encodec-decoded audio:
  1. High-frequency rolloff  — suppresses Encodec quantisation noise above cutoff
  2. LUFS normalisation      — loudness target for consistent output levels
     (falls back to RMS normalisation if pyloudnorm is not installed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# HF rolloff
# ---------------------------------------------------------------------------

def apply_hf_rolloff(
    wav: np.ndarray,
    sr: int,
    cutoff_hz: float = 10_000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to attenuate content above cutoff_hz.

    Encodec's high-frequency codebooks (cb4-cb7) often produce quantisation
    artefacts above 10 kHz.  A gentle rolloff here removes audible noise
    without affecting the melodic or timbral content.

    Args:
        wav:        Audio array, shape [T] or [T, C], float32.
        sr:         Sample rate in Hz.
        cutoff_hz:  -3dB point.  10 kHz is conservative; 12 kHz is subtler.
        order:      Filter order.  4 gives ~24 dB/oct rolloff.

    Returns:
        Filtered audio, same shape and dtype as input.
    """
    nyq = sr / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.9999)  # must be < 1
    sos = butter(order, norm_cutoff, btype="low", output="sos")

    dtype_in = wav.dtype
    wav_f = wav.astype(np.float64)

    if wav_f.ndim == 1:
        out = sosfiltfilt(sos, wav_f)
    else:
        out = np.stack([sosfiltfilt(sos, wav_f[:, c]) for c in range(wav_f.shape[1])], axis=1)

    return out.astype(dtype_in)


# ---------------------------------------------------------------------------
# Loudness normalisation
# ---------------------------------------------------------------------------

def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def apply_lufs_normalization(
    wav: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    peak_db: float = -1.0,
) -> np.ndarray:
    """
    Normalise to target integrated loudness (LUFS).

    Uses pyloudnorm when available (proper ITU-R BS.1770 measurement).
    Falls back to RMS normalisation targeting an equivalent level.

    Args:
        wav:          Audio array [T] or [T, C], float32/float64.
        sr:           Sample rate.
        target_lufs:  Target loudness in LUFS.  -14 is streaming standard
                      (Spotify/YouTube); -18 is more conservative.
        peak_db:      Hard limiter ceiling after gain, default -1 dBFS.

    Returns:
        Normalised audio, same shape and dtype.
    """
    dtype_in = wav.dtype
    wav_f = wav.astype(np.float64)

    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyloudnorm expects [T] (mono) or [T, C] (multi-channel)
        arr = wav_f if wav_f.ndim == 2 else wav_f
        loudness = float(meter.integrated_loudness(arr))
        if not np.isfinite(loudness):
            raise ValueError("Non-finite loudness measurement; falling back to RMS")
        gain_db = target_lufs - loudness
        out = wav_f * _db_to_amp(gain_db)
    except Exception:
        # RMS fallback: target_lufs ≈ target_rms_db - 3 dB empirically
        target_rms_db = target_lufs + 3.0
        rms = float(np.sqrt(np.mean(wav_f ** 2)))
        if rms <= 0.0:
            return wav
        target_rms = _db_to_amp(target_rms_db)
        out = wav_f * (target_rms / rms)

    # Hard peak limiter
    peak_amp = _db_to_amp(peak_db)
    out = np.clip(out, -peak_amp, peak_amp)
    return out.astype(dtype_in)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def postprocess_wav(
    in_path: str | Path,
    out_path: str | Path,
    *,
    hf_cutoff_hz: float = 10_000.0,
    hf_filter_order: int = 4,
    target_lufs: float = -14.0,
    peak_db: float = -1.0,
) -> None:
    """
    Read a WAV file, apply HF rolloff + loudness normalisation, overwrite it.

    Args:
        in_path:         Input WAV path (read-only; pass same path as out_path
                         to do in-place).
        out_path:        Where to write the processed WAV.
        hf_cutoff_hz:    Low-pass cutoff frequency (Hz).
        hf_filter_order: Butterworth filter order.
        target_lufs:     Integrated loudness target (LUFS).
        peak_db:         Peak ceiling after gain (dBFS).
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav, sr = sf.read(str(in_path), dtype="float32", always_2d=False)

    wav = apply_hf_rolloff(wav, sr, cutoff_hz=hf_cutoff_hz, order=hf_filter_order)
    wav = apply_lufs_normalization(wav, sr, target_lufs=target_lufs, peak_db=peak_db)

    sf.write(str(out_path), wav, sr, subtype="PCM_16")
