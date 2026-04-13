# Sangeet — Carnatic Music Generation Pipeline (Encodec + Transformer)

End-to-end, config-driven pipeline for training a MusicGen-style autoregressive model on **Carnatic** recordings using **Dunya** + **Saraga-style** metadata.

## Dataset layout (expected)

```
dataset/
  carnatic/
    Album/
      Song/
        song.json   # must contain "mbid" (+ optional raaga/taala/artists fields)
        song.mp3    # downloaded by the pipeline
```

Audio preprocessing and tokens are written outside `dataset/` by default (see `configs/*.yaml`).

## Prerequisites (Windows)

- Python 3.11+
- `ffmpeg` + `ffprobe` available on `PATH` (used for MP3 decoding/segmentation)
- PyTorch + CUDA (recommended)

## Setup

Install PyTorch first (pick the CUDA build that matches your GPU/driver):
```
# See: https://pytorch.org/get-started/locally/
```

```
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Dunya authentication

Set your token in an environment variable (do **not** hardcode it in scripts/configs):

PowerShell:
```
$env:DUNYA_TOKEN="YOUR_TOKEN"
```

## Pipeline

### 1) Download MP3 from Dunya

If you already have `dataset/carnatic/**/song.json` with MBIDs:
```
python preprocessing\\download_dunya_carnatic.py --config configs\\download_carnatic.yaml
```

To **crawl all Carnatic recordings** from Dunya and build the folder structure:
```
python preprocessing\\download_dunya_carnatic.py --config configs\\download_carnatic.yaml --mode crawl
```

### 2) Preprocess audio (MP3 → WAV, normalize, segment)

Writes segmented WAV files and a JSONL manifest:
```
python preprocessing\\preprocess_carnatic_audio.py --config configs\\preprocess.yaml
```

Outputs:
- `data/processed/wav16k/<mbid>/seg_XXXXX.wav`
- `data/processed/segments_manifest.jsonl`

### 3) Encodec tokenization (24kHz model)

```
python preprocessing\\tokenize_carnatic_encodec.py --config configs\\tokenize_encodec_24khz.yaml
```

Outputs:
- token files: `data/tokens/encodec_24khz_bw6/<mbid>/seg_XXXXX.npz`
- tokens manifest: `data/tokens/encodec_24khz_bw6/manifest.jsonl`

### 4) Train autoregressive Transformer LM

```
python training\\train_carnatic_lm.py --config configs\\train_carnatic_small.yaml
```

Checkpoints:
- `runs/carnatic_small/checkpoints/step_*.pt`
- `runs/carnatic_small/checkpoints/latest.pt`

Resume:
```
python training\\train_carnatic_lm.py --config configs\\train_carnatic_small.yaml --resume runs\\carnatic_small\\checkpoints\\latest.pt
```
python training\train_carnatic_lm.py --config configs\train_hindustani_small.yaml --resume runs\hindustani_small\checkpoints\latest.pt

### 5) Inference (generate → decode to WAV)

Edit `configs/infer.yaml` (raga/tala/artist/text + duration), then:
```
python inference\\generate_carnatic.py --config configs\\infer.yaml
```

## Notes

- Conditioning uses `raga`, `tala`, `artist` (from manifests / `song.json`) via cross-attention.
- Optional `text` conditioning is supported via a minimal UTF-8 byte tokenizer.
- Segment length strongly affects sequence length and training memory; start with ~10s segments (`configs/preprocess.yaml`).
