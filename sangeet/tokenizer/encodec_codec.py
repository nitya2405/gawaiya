from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pyexpat import model
from typing import Any

import numpy as np
import torch


class EncodecNotInstalledError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodecConfig:
    model: str = "24khz"  # 24khz
    bandwidth: float = 6.0
    device: str = "cuda"
    use_normalize: bool = False


def _require_encodec():
    try:
        from encodec import EncodecModel  # noqa: F401
        from encodec.utils import convert_audio  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise EncodecNotInstalledError(
            "Missing dependency `encodec`. Install with `pip install encodec`."
        ) from e


def load_encodec_model(cfg: EncodecConfig) -> Any:
    _require_encodec()
    from encodec import EncodecModel

    if cfg.model != "24khz":
        raise ValueError(f"Unsupported Encodec model: {cfg.model}")

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(float(cfg.bandwidth))
    # We loudness-normalize separately; keeping model normalization off avoids needing to store scales.
    try:
        model.normalize = bool(cfg.use_normalize)
    except Exception:
        pass

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def encode_wav_file(model: Any, wav_path: str | Path) -> dict[str, Any]:
    _require_encodec()
    from encodec.utils import convert_audio

    wav_path = Path(wav_path)
    wav, sr = _load_wav_torch(wav_path)

    # Skip empty / corrupted audio
    if wav.numel() == 0 or wav.shape[-1] < sr * 0.1:
        raise ValueError(f"Empty or too-short audio: {wav_path}")


    # device = next(model.parameters()).device
    # wav = wav.to(device)
    # wav = convert_audio(wav, sr, model.sample_rate, model.channels)

    # Force resample on CPU (torchaudio resampler is CPU-only on Windows)
    wav = wav.cpu()

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)

    # Move back to model device (GPU if available)
    device = next(model.parameters()).device
    wav = wav.to(device)


    encoded_frames = model.encode(wav)  # list[(codes, scale)]

    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)  # [B, n_q, T]
    codes = codes[0].contiguous()

    # Prefer model attributes when available.
    n_codebooks = int(codes.shape[0])
    n_frames = int(codes.shape[1])

    codebook_size = None
    for attr in ("codebook_size", "bins"):
        v = getattr(getattr(model, "quantizer", None), attr, None)
        if isinstance(v, int):
            codebook_size = int(v)
            break
    if codebook_size is None:
        codebook_size = int(codes.max().item() + 1)

    # Estimate frame rate if not present.
    frame_rate = getattr(model, "frame_rate", None)
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        frame_rate = float(n_frames) / (float(wav.shape[-1]) / float(model.sample_rate))

    return {
        "codes": codes.to(torch.int16).cpu().numpy().astype(np.int16),  # [n_codebooks, n_frames]
        "sample_rate": int(model.sample_rate),
        "channels": int(model.channels),
        "bandwidth": float(getattr(model, "bandwidth", 0.0) or 0.0),
        "n_codebooks": n_codebooks,
        "codebook_size": int(codebook_size),
        "frame_rate": float(frame_rate),
    }


def _load_wav_torch(path: Path) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required to load WAV files. `pip install soundfile`.") from e

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    # soundfile uses [T,C]; convert to torch [B=1,C,T]
    audio_t = torch.from_numpy(audio.T).unsqueeze(0).contiguous()
    return audio_t, int(sr)


@torch.inference_mode()
def decode_codes_to_wav(
    model: Any,
    *,
    codes: np.ndarray,
    out_wav_path: str | Path,
    sample_rate: int | None = None,
) -> None:
    _require_encodec()
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    codes_t = torch.from_numpy(codes.astype(np.int64)).to(device)  # [n_codebooks, T]
    codes_t = codes_t.unsqueeze(0)  # [B, n_codebooks, T]

    # Encodec expects a list of frames (codes, scale). With normalize=False, scale can be None.
    wav = model.decode([(codes_t, None)])
    wav = wav[0].detach().cpu()  # [C, T]

    sr = int(sample_rate or model.sample_rate)
    audio = wav.transpose(0, 1).numpy()  # [T, C]
    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required to save WAV files. `pip install soundfile`.") from e
    sf.write(str(out_wav_path), audio, sr, subtype="PCM_16")
