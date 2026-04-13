# Scale-Up Plan — Sangeet AI

> Hindustani Classical Music Generation: from working ML pipeline to shippable product.

---

## Current State (April 2026)

### What works
- Encodec (24kHz, 6kbps) tokenization over 15,697 segments (~43 hours of audio)
- Transformer LM (512d, 12 layers) trained to 155k steps with CFG
- 63 ragas, 10 talas in vocabulary
- Classifier-Free Guidance (CFG scale 5.0) producing raga-accurate output
- 4-codebook generation with 12-13s coherent context window
- Stitched output up to any length via crossfades
- Post-processing: HF rolloff + LUFS normalization
- CLI: `generate_music.py --raga "Kalyāṇ" --duration 60`

### Known limitations
- Vocals are garbled (Encodec not vocal-specific)
- No compositional structure (alap/jor/jhala)
- Tabla drops after ~12s (context window exceeded)
- Generation: ~30-60 seconds wall time for 60s output on RTX 3050

---

## Phase 1 — Ship a Working Demo (~10 working days)

**Goal:** Real users can open a URL, pick a raga, and hear 30-60 seconds of music back.

### 1.1 Backend (FastAPI)

**Stack:** Python, FastAPI, Celery, Redis

**Endpoints:**
```
POST   /api/generate          → { job_id }
GET    /api/job/{id}           → { status, progress, queue_position, error }
GET    /api/audio/{id}         → stream MP3 (chunked)
GET    /api/ragas              → list with metadata (name, thaat, time, mood)
GET    /api/talas              → list of available talas
GET    /api/feedback/{id}      → submit 👍/👎 (query param: value=1 or -1)
WS     /api/ws/{job_id}        → real-time progress stream
```

**Key design decisions:**
- Model loads once at startup, stays in GPU memory permanently
- Single Celery worker (one job at a time — GPU constraint)
- Redis for job state: status, progress %, queue position, output path, error
- Redis token bucket rate limiting: 10 generations per IP per hour (not hard block — CGNAT-safe)
- Queue position surfaced to frontend ("you are #3 in queue")
- Generated files stored as MP3 in `outputs/api/` with UUID filenames
- ffmpeg transcodes WAV → MP3 192kbps at end of Celery task (WAV deleted after)
- Job TTL: 1 hour (auto-cleanup)
- Audio served in chunks from FastAPI (not full-file load into memory)

**Generation parameters exposed to API:**
- `raga` (string, validated against vocab)
- `tala` (string, validated against vocab)
- `duration_sec` (int, 6–60 seconds — default **12s**, stitching kicks in above 12s)
- `cfg_scale` (float, 3.0–7.0, default 5.0)
- `n_codebooks` (int, 2/4/8, default 4)

**File structure:**
```
backend/
  main.py           ← FastAPI app, all routes
  worker.py         ← Celery task (generation + ffmpeg transcode)
  schemas.py        ← Pydantic request/response models
  config.py         ← env vars, paths, model config
  model_cache.py    ← singleton model loader (load once, reuse)
  raga_meta.py      ← hardcoded raga metadata JSON (thaat, time, mood)
  rate_limit.py     ← Redis token bucket implementation
  feedback.jsonl    ← append-only feedback log (one JSON record per line)
tests/
  test_api.py       ← one smoke test: POST /generate → job completes → MP3 playable
requirements-api.txt
```

**Feedback storage:**
Each 👍/👎 appends one line to `feedback.jsonl`:
```json
{"timestamp": "2026-04-13T12:00:00Z", "job_id": "uuid", "raga": "Kalyāṇ", "tala": "Tīntāl", "duration_sec": 45, "rating": 1}
```
Redis list (`RPUSH feedback_queue`) buffers writes; a background thread flushes to disk every 60s. This survives a restart without losing in-flight feedback.

---

### 1.2 Frontend (React + Vite)

**Stack:** React, Vite, TailwindCSS, WaveSurfer.js, Axios

**Single page layout:**
```
┌──────────────────────────────────────────────────────────────┐
│  🎵 Sangeet AI              Hindustani Classical Generator    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Raga    [ Kalyāṇ                              ▼ ]          │
│            Evening raga · Kalyan thaat · evokes longing      │
│                                                              │
│   Tala    [ Tīntāl                              ▼ ]          │
│                                                              │
│   Duration  ●──────────────────────────  12s  ← default      │
│             6s                          60s                  │
│   ✦ Best quality · single continuous generation              │
│                                                              │
│   — — — — — — — — — — — — — — — — — — — — — — — — — — —      │
│   (user drags to 35s)                                        │
│   Duration  ○────────────────●──────────  35s                │
│   ⚠ Beyond 12s, clips are stitched with a crossfade.         │
│     Transitions may be audible. Quality varies.              │
│                                                              │
│   ▸ Advanced                                                 │
│     CFG Scale   ○──────●───  5.0                             │
│     Codebooks   ○──────●───  4                               │
│                                                              │
│         [ Generate Music ]                                   │
│                                                              │
│   You are #2 in queue · estimated wait ~40s                  │
│   ──────── Generating clip 3 / 4 ──────────── 60% ──────     │
│                                                              │
│   ▶ ──────────────────────────────────── 0:23 / 0:45         │
│   [ ⬇ Download MP3 ]  [ 🔁 Regenerate ]  [ 👍 ]  [ 👎 ]       │
│                                                              │
│   Recent generations ─────────────────────────────────────   │
│   Bhairav 30s  ·  Tīntāl 45s  ·  Yaman Kalyāṇ 60s           │
└──────────────────────────────────────────────────────────────┘
```

**Components:**
- `RagaSelector` — searchable dropdown, 63 options, shows 1-line context below
- `TalaSelector` — dropdown, 10 options
- `DurationSlider` — 6s to 60s, **default 12s** (single coherent clip, no stitching); shows "✦ Best quality" badge at ≤12s; shows ⚠ crossfade warning when >12s
- `AdvancedControls` — collapsible: CFG scale, codebook count
- `GenerateButton` — disabled while job running, shows queue position when waiting
- `ProgressBar` — WS-driven with polling fallback (try WS → if fail → poll /job/{id} every 2s)
- `AudioPlayer` — WaveSurfer.js waveform + play/pause/seek
- `DownloadButton` — fetches `/api/audio/{id}` as MP3
- `FeedbackButtons` — 👍/👎, calls `/api/feedback/{id}`, one-shot (disabled after click)
- `GenerationHistory` — last 5 outputs in localStorage with raga/tala/duration labels

**File structure:**
```
frontend/
  src/
    components/
      RagaSelector.jsx
      TalaSelector.jsx
      DurationSlider.jsx
      AudioPlayer.jsx
      ProgressBar.jsx
      FeedbackButtons.jsx
      GenerationHistory.jsx
    hooks/
      useGeneration.js    ← POST → WS (fallback: poll) → audio URL
      useVocab.js         ← fetches /api/ragas and /api/talas
    data/
      raga_meta.json      ← mirrors backend raga_meta.py for UI display
    App.jsx
    main.jsx
  index.html
  vite.config.js
  tailwind.config.js
  package.json
```

---

### 1.3 WebSocket + Polling Fallback

`useGeneration.js` connection strategy:
```
POST /api/generate → job_id
  → try WebSocket /api/ws/{job_id}
      if WS connects: drive progress from WS messages
      if WS fails (timeout 3s or error): fall back to polling
  → polling: GET /api/job/{id} every 2s until status=done|failed
  → on done: set audio URL to /api/audio/{id}
```

This handles corporate proxies, mobile connections, and any environment that silently blocks WebSockets.

**Audio src timing — critical:** the `<audio>` element's `src` must be set only *after* `status=done` is confirmed, never during generation. Setting src to a file still being written causes silent or truncated playback on slow connections. The flow is strictly: job completes → file closed → then frontend sets src.

---

### 1.4 Local Dev Setup

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  React (Vite)   │────▶│  FastAPI     │────▶│  Celery      │
│  localhost:5173 │     │  :8000       │     │  Worker(GPU) │
└─────────────────┘     └──────┬───────┘     └──────┬───────┘
                               │                    │
                               └────────┬───────────┘
                                        │
                                   ┌────▼────┐
                                   │  Redis  │
                                   │  :6379  │
                                   └─────────┘
```

**Run order:**
```bash
redis-server                                    # terminal 1
celery -A backend.worker worker --loglevel=info # terminal 2 (GPU)
uvicorn backend.main:app --reload               # terminal 3
npm run dev  (inside frontend/)                 # terminal 4
```

---

## Phase 1.5 — ML Upgrade (runs in background during Phase 1)

**Goal:** Fix the single highest-ROI ML problem before any real user outreach.

### Delay pattern / MusicGen-style interleaving

The current flat AR over 8 codebooks is the root cause of tabla dropout and high-codebook noise. MusicGen's delay pattern generates codebooks with offset steps, giving each codebook independent context:

```
Current (flat):   cb0 cb1 cb2 cb3 cb4 cb5 cb6 cb7 | cb0 cb1 cb2 ...
Delay pattern:    cb0                               | cb0
                      cb1                           |     cb1
                          cb2                       |         cb2
```

**What this requires:**
1. Re-tokenize all 15,697 segments with delay pattern encoding
2. Retrain transformer from scratch (~120k steps, ~7 days on RTX 3050)
3. Update `generate_music.py` decoding accordingly

**Expected improvements:**
- Tabla stays coherent across entire output (no more drops)
- High-codebook static dramatically reduced
- Longer effective coherence per clip

**Model interface contract — drop-in requirement:**
The Phase 1.5 checkpoint must be swappable into the Phase 1 backend by changing a single line in `backend/config.py`:
```python
MODEL_CHECKPOINT = "runs/hindustani_delay/checkpoints/latest.pt"
```
This means the new model must expose identical `generate()` signature and token spec. Any delay pattern decoding changes go inside `generate_music.py` / `model_cache.py` — the API layer never changes. Validate this contract before starting the retrain so the swap is genuinely one file path change.

**This trains while Phase 1 ships. Merges into Phase 2.**

---

## Phase 2 — Quality + Reach (Month 2)

**Goal:** Better audio, shareable links, broader raga coverage.

### ML
- Merge delay pattern model from Phase 1.5
- Experiment with 48kHz Encodec (requires cloud GPU)
- Add "style preset" conditioning tokens: Alap / Bandish / Tarana
  - Controls tempo character and whether tabla appears
- Begin using 👍/👎 feedback data to weight training samples

### Product
- Shareable UUID links per generation (GET /share/{id})
- Spectrogram view toggle in AudioPlayer
- Raga of the day on homepage
- Social preview meta tags (og:audio, og:image of waveform)

---

## Phase 3 — Production (Month 3+)

**Goal:** Public-facing, handles concurrent users, monetizable.

### Infrastructure
```
Vercel (frontend)
    ↓
Railway / Render (FastAPI)
    ↓
RunPod A4000 16GB (Celery GPU worker — always-on, not serverless)
    ↓
Redis Cloud + Cloudflare R2 (audio storage, pre-signed URLs)
```

**Critical:** GPU worker must be always-on (not serverless). Cold model load = 30-60 seconds. FastAPI returns a pre-signed R2 URL directly — no audio bytes flow through the API server.

### Cloud GPU unlocks
On A4000 (16GB VRAM):
- 8 codebooks cleanly, no static
- `max_seq_len: 8192` → ~13s coherent without stitching
- Larger model: 768d/16L or 1024d/24L
- Estimated: $0.20/hr on-demand, ~$150/month always-on

### Additional features
- User accounts (Clerk or Supabase Auth)
- Per-user generation history
- API key access for developers
- Feedback data pipeline → active training data curation

---

## Immediate Next Steps (Phase 1)

| # | Task | Est. |
|---|------|------|
| 1 | `backend/main.py` — FastAPI skeleton, `/ragas`, `/talas` with metadata | 1 day |
| 2 | `backend/model_cache.py` — singleton loader | 0.5 day |
| 3 | `backend/raga_meta.py` — hardcoded thaat/time/mood for all 63 ragas | 0.5 day |
| 4 | `backend/worker.py` — Celery task: generate → ffmpeg → MP3 | 1 day |
| 5 | `backend/main.py` — `/generate`, `/job/{id}`, `/audio/{id}`, `/feedback/{id}` | 1 day |
| 6 | `backend/rate_limit.py` — Redis token bucket (10/hr/IP) + queue position | 0.5 day |
| 7 | WebSocket progress endpoint + chunked audio streaming | 0.5 day |
| 8 | React scaffold (Vite + Tailwind) | 0.5 day |
| 9 | `RagaSelector` with inline metadata + `TalaSelector` + `DurationSlider` (6-60s, default 12s, crossfade warning >12s) | 1 day |
| 10 | `useGeneration.js` — WS with polling fallback | 1 day |
| 11 | `AudioPlayer` (WaveSurfer.js) + `FeedbackButtons` + `DownloadButton` | 1 day |
| 12 | `GenerationHistory` (localStorage) | 0.5 day |
| 13 | `tests/test_api.py` — basic endpoint smoke tests | 0.5 day |
| 14 | End-to-end: browser → generate → play → download → feedback | 0.5 day |

**Total Phase 1 estimate: ~10 working days.**

---

## Repo Structure (Target)

```
sangeet/
├── backend/              ← FastAPI + Celery (Phase 1)
├── frontend/             ← React + Vite (Phase 1)
├── tests/                ← API smoke tests
├── docs/                 ← Plans, architecture notes
│   └── scale_up_plan.md
├── eval/                 ← Evaluation scripts
├── inference/            ← generate_hindustani.py
├── training/             ← train_carnatic_lm.py
├── preprocessing/        ← tokenization pipeline
├── configs/              ← YAML configs
├── sangeet/              ← Core Python package
├── outputs/              ← All generated audio (gitignored)
│   ├── v1/
│   ├── v2/
│   ├── baseline/
│   ├── cfg_eval/
│   ├── cb_depth/
│   ├── music/
│   └── api/              ← API-generated MP3s (Phase 1)
├── runs/                 ← Checkpoints (gitignored)
├── generate_music.py     ← CLI entry point
└── requirements.txt
```

---

## Dream Goal Gap Analysis

| Dimension | April 2026 | Dream |
|---|---|---|
| Raga accuracy | ~65%, conditioned correctly | >90%, compositional structure |
| Vocals | Garbled (Encodec limit) | Clear vowels, meend, gamak |
| Structure | Random phrases | Alap → Jor → Jhala |
| Tabla | Drops after 12s | Consistent tala cycles |
| Max coherent duration | 12s (4 codebooks) | 5-10 minutes |
| UI | CLI | Web app |
| Users | Solo | Public |
| Hardware | RTX 3050 4GB | A100 40GB |
| Model size | 512d/12L (~80M params) | 1024d/24L (~600M params) |

**The ML and the product are parallel tracks. Phase 1 ships on current quality. Phase 1.5 ML trains in the background. They merge in Phase 2.**
