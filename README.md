<p align="center">
  <img src="docs/assets/hero-banner.png" alt="IOGHAL — Pluggable Local GPU Voice Runtime" width="100%" />
</p>

<h1 align="center">IOGHAL</h1>

<p align="center">
  <strong>Pluggable local GPU Push-to-Talk voice runtime powered by VoxCPM2, Muse-Glimmer / Ollama, and Whisper Large v3.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-1f5c99.svg" alt="License: Apache 2.0" /></a>
  <img src="https://img.shields.io/badge/Python-3.12-276bb0.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Audio-48_kHz_Streaming_PCM-e6261d.svg" alt="48 kHz Streaming PCM" />
  <img src="https://img.shields.io/badge/Frontend-React_19_+_WebAudio-1f5c99.svg" alt="React 19 + WebAudio" />
</p>

---

IOGHAL runs a complete local speech-to-speech loop on a single GPU. It captures microphone audio, transcribes speech with FP8 + CUDA-graphed Whisper Large v3, grounds current/encyclopedic queries with live Wikipedia & DuckDuckGo Lite search, streams replies sentence by sentence from Meta Muse-Glimmer-30B (via direct CUDA 13 `llama-server` with DFlash-15 speculative decoding and Zero-Think / Low-Think routing) or local Ollama models, and synthesizes 48 kHz mono cloned speech in real time with Mega-Step CUDA-graphed OpenBMB VoxCPM2 (with optional merged LoRA weights).

Every core stage sits behind a runtime-checkable Python `Protocol`. You can run IOGHAL as a standalone React web console, a PyQt6 desktop application, or a terminal interface, or embed its pipeline, FastAPI router, and React optical eye component directly into a larger application.

---

## System Architecture

<p align="center">
  <img src="docs/assets/architecture-diagram.png" alt="IOGHAL Streaming Voice Pipeline Architecture" width="100%" />
</p>

```text
+---------------------------------------------------------------------------------------------------+
|                                   CLIENT / HOST APPLICATION                                       |
|  +---------------------------+   +---------------------------+   +-----------------------------+  |
|  | React Web Console (Vite)  |   | PyQt6 Desktop / CLI App   |   | Host App (Python / FastAPI) |  |
|  | <HalEye /> + HalAudioPlay |   | HalEyeWidget + PortAudio  |   | HalPipeline / HalRouter     |  |
|  +-------------+-------------+   +-------------+-------------+   +--------------+--------------+  |
+----------------|-------------------------------|--------------------------------|-----------------+
                 | JSON Control +                | Direct Call                    | Protocol / WS
                 | Binary PCM [u32 tid | f32[]]  |                                |
                 v                               v                                v
+---------------------------------------------------------------------------------------------------+
|                                      IOGHAL CORE RUNTIME                                          |
|                                                                                                   |
|   +------------------------+          +-------------------------------------------------------+   |
|   | AudioInput Protocol    |          | HalRuntime (Shared GPU Lock + Per-Conn Isolation)     |   |
|   | - PipeWire MicRecorder |          |  - Single-writer async WebSocket queue (non-blocking) |   |
|   | - Browser WebAudio Mic |          |  - Client wire turn_id + internal cancel generation   |   |
|   +-----------+------------+          |  - Exclusive host-mic ownership (loopback-guarded)    |   |
|               | 16 kHz mono float32   +---------------------------+---------------------------+   |
|               v                                                   |                               |
|   +---------------------------------------------------------------v---------------------------+   |
|   |                                     HalPipeline.run_turn()                                |   |
|   |                                                                                           |   |
|   |  +----------------------+      +---------------------------+      +--------------------+  |   |
|   |  | STTProvider          | text | BrainProvider             | sent | TTSProvider        |  |   |
|   |  | WhisperSTT (CUDA)    +----->| OllamaBrain (Ollama Stream) +----->| HalVoiceEngine     |  |   |
|   |  | - Local-first cache  |      | - LoreIndex recall        |      | - VoxCPM2 48 kHz   |  |   |
|   |  | - Voiced energy gate |      | - Think-tag stripper      |      | - Prompt cache     |  |   |
|   |  | - Domain prompt IDs  |      | - Abbreviation-safe split |      | - Phrase PCM cache |  |   |
|   |  +----------+-----------+      +---------------------------+      +---------+----------+  |   |
|   |             ^                                                               ^             |   |
|   |             +------------------- Shared threading.Lock (gpu_lock) ----------+             |   |
|   +-----------------------------------------------------------------------------+-------------+   |
+---------------------------------------------------------------------------------|-----------------+
                                                                                  |
                                   48 kHz Float32 PCM Chunks + HalEvent Stream    |
                                   (deferred audio_start, barge-in cancel, RTF)   |
                                                                                  v
                                              [ Client Audio Playback & Reactive Optical Eye ]
```

---

## Pluggable Interfaces

<p align="center">
  <img src="docs/assets/plugin-interfaces.png" alt="IOGHAL Pluggable Interface Architecture" width="100%" />
</p>

IOGHAL separates transport, UI, and model inference into small interfaces in [src/hal9000/interfaces.py](src/hal9000/interfaces.py):

| Interface | File | Default Implementation | Purpose |
|---|---|---|---|
| `STTProvider` | `src/hal9000/interfaces.py` | `WhisperSTT` (`stt.py`) | FP8 + CUDA-graphed Whisper Large v3 converting mono float32 audio to text (auto-resampling to 16 kHz). |
| `BrainProvider` | `src/hal9000/interfaces.py` | `OllamaBrain` (`brain.py`) | Dual-backend (`llama-server` SSE & Ollama) streaming brain with Zero-Think/Low-Think routing, KV prefix reuse, and lore recall. |
| `TTSProvider` | `src/hal9000/interfaces.py` | `HalVoiceEngine` (`tts.py`) | Mega-Step CUDA-graphed VoxCPM2 streaming 48 kHz mono PCM with overlapped AudioVAE, prompt cache, and optional merged LoRA. |
| `AudioInput` | `src/hal9000/interfaces.py` | `MicRecorder` (`audio.py`) | Captures push-to-talk audio with bounded duration and live RMS telemetry. |
| `HalPipeline` | `src/hal9000/pipeline.py` | `HalPipeline` | Runs one cancellable STT -> Live Web Search -> concurrent LLM \|\| TTS turn and emits typed `HalEvent` objects. |
| `create_hal_router` | `src/hal9000/server.py` | FastAPI `APIRouter` | Mounts `/api/health` and `/ws` into any existing FastAPI server. |
| `<HalEye />` / `HalAudioPlayer` | `web/src/index.ts` | React 19 + WebAudio | Embeddable reactive optical lens and framed-PCM audio playback engine. |

### 1. Plug into a Python Application

Inject any custom `STTProvider`, `BrainProvider`, or `TTSProvider` into `HalPipeline`:

```python
import threading
from hal9000 import HalEvent, HalPipeline, HalVoiceEngine, OllamaBrain, WhisperSTT

pipeline = HalPipeline(
    stt=WhisperSTT(),
    brain=OllamaBrain(model="lfm3:1.7b", user_name="Charles"),
    tts=HalVoiceEngine(reference_wav="voices/hal9000_ref.wav", inference_timesteps=6),
)

def on_event(event: HalEvent) -> bool:
    if event.kind == "hal_partial":
        print("HAL:", event.payload["text"])
    elif event.kind == "audio_chunk" and event.pcm is not None:
        # Forward 48 kHz mono float32 PCM to your host application's audio sink
        pass
    return True

result = pipeline.run_turn(
    turn_id=1,
    cancel_event=threading.Event(),
    on_event=on_event,
    user_text="Good afternoon, HAL. Confirm all systems.",
)
```

### 2. Mount into an Existing FastAPI Server

Attach IOGHAL to any FastAPI service at a custom URL prefix with injected configuration or custom providers:

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from hal9000 import HalConfig, HalRuntime, create_hal_router

runtime = HalRuntime(
    HalConfig(
        model="lfm3:1.7b",
        steps=6,
        user_name="Charles",
        allowed_origins=("http://localhost:3000",),
    )
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.set_loop(asyncio.get_running_loop())
    runtime.start_background_load()
    try:
        yield
    finally:
        runtime.shutdown()

app = FastAPI(lifespan=lifespan)
app.include_router(create_hal_router(runtime, prefix="/hal"))
```

### 3. Embed the React Optical Eye and Audio Player

Import the standalone canvas widget and framed PCM player from [web/src/index.ts](web/src/index.ts):

```tsx
import { useRef } from 'react';
import { HalEye, HalAudioPlayer, decodeFramedPcm } from './web/src';
```

`HalAudioPlayer` handles prebuffering, underrun recovery, mute state, anti-pop gain ramps on barge-in, autoplay suspension callbacks, and one-click replay of the last completed transmission.

---

## WebSocket Wire Protocol

The `/ws` endpoint uses JSON control messages and framed binary PCM chunks over a single ordered per-connection writer queue.

### Client to Server (JSON)

| Type | Fields | Behavior |
|---|---|---|
| `ptt_start` | `capture_id` | Acquires loopback host microphone and starts 16 kHz capture. |
| `ptt_stop` | `turn_id`, `capture_id` | Stops host microphone and runs STT -> Brain -> TTS under `turn_id`. |
| `audio_b64` | `turn_id`, `pcm_b64`, `sample_rate` | Submits browser-captured float32 PCM (validated up to 30 s, 8–96 kHz). |
| `text` | `turn_id`, `text` | Runs Brain -> TTS directly (up to 4,000 characters). |
| `interrupt` | `turn_id`, `silent?` | Cancels the active turn and stops local capture. |
| `set_model` | `model` | Switches the active `llama-server` or Ollama model for the current connection. |
| `set_steps` | `steps` | Updates per-connection VoxCPM2 diffusion steps (`2` to `20`). |
| `reset` | `turn_id` | Cancels active work and clears the connection's conversation history. |

### Server to Client

- **JSON messages:** `system_status`, `state`, `capture_failed`, `capture_limit`, `mic_level`, `transcript`, `audio_start`, `hal_partial`, `metrics`, `audio_end`, `cleared`, and `error`.
- **Binary PCM frames:** Every binary WebSocket frame starts with a 4-byte little-endian `uint32` `turn_id` header followed by little-endian `float32` mono PCM samples at `sample_rate` Hz. The client drops any frame whose header does not match its active `turn_id`.

---

## Requirements

- Linux with an NVIDIA CUDA GPU (tested on Python 3.12 and PyTorch CUDA 13.0; Blackwell/Ada GPUs automatically enable native FP8 + CUDA Graphs)
- [uv](https://docs.astral.sh/uv/) and Node.js 20+
- [Ollama](https://ollama.com/) running locally with `muse-glimmer:30b-q4_K_M-dflash` (served directly via `./run_glimmer_server.sh` with DFlash-15) and/or local Ollama models (`gemma4:26b`, `qwen3.5:9b-q8_0`, `lfm3:1.7b`)
- [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM) checked out at `../VoxCPM` (editable local source configured in [pyproject.toml](pyproject.toml))
- A 48 kHz mono reference WAV clip at `voices/hal9000_ref.wav` (or set `HAL_REF_WAV` / `--ref-wav`; see [voices/README.md](voices/README.md)), plus optional LoRA weights at `voices/hal9000_lora/` (or `HAL_LORA_DIR`)

---

## Quick Start

### 1. Install Dependencies and Build the Frontend

```bash
# Install Python dependencies into .venv
uv sync

# Build the React web console into web/dist
cd web && npm install && npm run build && cd ..
```

### 2. Pull Local Models (Muse-Glimmer-30B + DFlash or Ollama)

```bash
# Recommended: Meta Muse-Glimmer-30B with DFlash-15 speculative decoding
ollama pull muse-glimmer:30b-q4_K_M-dflash

# Optional Ollama fallback / fast models
ollama pull qwen3.5:9b-q8_0
ollama pull lfm3:1.7b
```

### 3. Add a Reference Voice Clip

Place a 5 to 12 second 48 kHz mono WAV clip at `voices/hal9000_ref.wav` (with an optional matching `voices/hal9000_ref.txt` transcript for Ultimate Cloning), or export `HAL_REF_WAV=/path/to/voice.wav`.

### 4. Run Muse-Glimmer (`llama-server`) and the Web Console

```bash
# Terminal 1 (recommended): launch Muse-Glimmer-30B + DFlash-15 on :8080 using templates/glimmer_hal_v2.jinja
./run_glimmer_server.sh

# Terminal 2: launch the web console (auto-detects http://127.0.0.1:8080 when running)
./run_hal9000_web.sh --host 127.0.0.1 --port 8765 --steps 6
```

CLI / environment options for `hal9000-web` and `HalConfig`:
- `--llama-server-url` (`HAL_LLAMA_SERVER_URL`): OpenAI-compatible `llama-server` base URL (auto-detected on `http://127.0.0.1:8080` by `run_hal9000_web.sh`)
- `--llm-backend` (`HAL_LLM_BACKEND`): `auto` (default), `llama_server`, or `ollama`
- `--thinking-mode` (`HAL_THINKING_MODE`): `auto` (Zero-Think for casual/lore, Low-Think for math/analysis/web search), `off`, or `low`
- `--reasoning-budget` (`HAL_REASONING_BUDGET`): Low-Think reasoning token cap (default: `64`)

Open `http://127.0.0.1:8765` in Brave or any Chromium/Firefox browser:

- **Hold Spacebar** (or press and hold the optical eye / PTT bar) to speak and release to send.
- **Quick-tap Spacebar** (`< 280 ms`) to latch listening; tap Spacebar again to send.
- **Press Escape** (or click `ESC`) at any time for instant barge-in cancellation.

### 5. Run Desktop (PyQt6) or Terminal CLI Mode

```bash
# PyQt6 desktop application
./run_hal9000.sh --steps 6

# Terminal Push-to-Talk mode
./run_hal9000.sh --cli --steps 6

# Standalone voice cloning + Whisper + Liquid verification script
uv run python test_clone.py --steps 6
```

---

## Repository Layout

```text
IOGHAL/
├── docs/assets/                # Hero banner, pipeline architecture, and interface diagrams
├── lore/                       # Optional local Markdown lore files for dynamic recall
├── src/hal9000/
│   ├── __init__.py             # Public Python SDK exports
│   ├── _cuda_ops.py            # Shared Blackwell/CUDA FP8, Split-K INT4 GEMV, and CUDA Graph helpers
│   ├── interfaces.py           # STTProvider, BrainProvider, TTSProvider, AudioInput, HalConfig
│   ├── pipeline.py             # Transport-agnostic concurrent STT -> Search -> LLM || TTS coordinator
│   ├── audio.py                # PipeWire MicRecorder and StreamingAudioPlayer
│   ├── stt.py                  # Local-first FP8 + CUDA-graphed WhisperSTT with speech activity gate
│   ├── brain.py                # Dual-backend llama-server/Ollama brain, Zero/Low-Think router, LoreIndex
│   ├── websearch.py            # Zero-API-key parallel Wikipedia & DuckDuckGo Lite live grounding
│   ├── tts.py                  # Mega-Step CUDA-graphed VoxCPM2 engine, overlapped AudioVAE, LoRA merge
│   ├── server.py               # FastAPI HalRuntime, create_hal_router(), WebSocket server
│   └── app.py                  # PyQt6 desktop GUI and terminal CLI application
├── templates/
│   └── glimmer_hal_v2.jinja    # Muse-Glimmer ATEM Jinja template for per-request Zero/Low-Think + KV reuse
├── specs/
│   ├── ioghal_protocol.qnt     # Executable Quint state-machine spec (multi-tab PTT & WS protocol)
│   ├── ioghal_protocol_test.qnt# Deterministic Quint protocol scenario tests
│   ├── glimmer_runtime.qnt     # Quint formal model of Glimmer runtime, KV cache, cancellation & CUDA graphs
│   └── glimmer_runtime_test.qnt# Counterexample & proof tests for glimmer_runtime.qnt
├── run_glimmer_server.sh       # Direct CUDA 13 llama-server launcher for Muse-Glimmer-30B + DFlash-15
├── run_hal9000_web.sh          # Web console launcher (auto-detects local llama-server on :8080)
├── tests/
│   └── test_hal9000.py         # Unit, protocol, framing, and concurrency test suite
├── voices/
│   └── README.md               # Voice reference setup instructions (WAV clips gitignored)
├── web/
│   └── src/
│       ├── index.ts            # Embeddable React component and audio client barrel export
│       ├── halClient.ts        # Framed PCM decoder, protocol types, HalAudioPlayer
│       ├── HalEye.tsx          # Reactive Canvas optical eye widget
│       ├── App.tsx             # Discovery One Push-to-Talk web console
│       └── styles.css          # Console layout and dark theme styles
├── pyproject.toml              # Package metadata, Apache-2.0 license, and direct dependencies
├── LICENSE                     # Apache License 2.0
└── test_clone.py               # End-to-end voice clone and round-trip verification script
```

---

## Verification

Run the Python test suite, linter, TypeScript production build, and Quint protocol checks:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests test_clone.py
cd web && npm run build && cd ..
quint test specs/ioghal_protocol_test.qnt --main=ioghal_protocol_test
quint test specs/glimmer_runtime_test.qnt --main=glimmer_runtime_test
quint run specs/ioghal_protocol.qnt --main=ioghal_protocol \
  --invariant=allInvariants --max-samples=10000 --max-steps=50
quint run specs/glimmer_runtime.qnt --main=glimmer_runtime \
  --invariant=allInvariants --max-samples=10000 --max-steps=50
```

---

## License

Released under the [Apache License 2.0](LICENSE).
