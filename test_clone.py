"""Standalone verification test: clone voice with VoxCPM2, verify via Whisper, and test Liquid."""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from hal9000.brain import OllamaBrain
from hal9000.stt import WhisperSTT
from hal9000.tts import (
    DEFAULT_PROJECT_DIR,
    MIN_DIFFUSION_STEPS,
    HalVoiceEngine,
    resolve_reference_wav,
)

DEFAULT_TEXT = (
    "Good afternoon. I am a HAL nine thousand computer. "
    "All circuits are operational, and VoxCPM voice synthesis is now online."
)


def _norm_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VoxCPM2 voice cloning, streaming, Whisper, and Liquid")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROJECT_DIR / "outputs" / "hal9000_test.wav",
    )
    parser.add_argument("--ref-wav", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--play", action="store_true", help="Play the synthesized clip aloud")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile warmup")
    args = parser.parse_args()

    if args.steps < MIN_DIFFUSION_STEPS:
        parser.error(f"--steps must be >= {MIN_DIFFUSION_STEPS}")

    print("1) Loading VoxCPM2 and caching voice prompt...")
    t0 = time.perf_counter()
    tts = HalVoiceEngine(
        reference_wav=resolve_reference_wav(args.ref_wav),
        optimize=args.compile,
        inference_timesteps=args.steps,
    )
    print(f"   Ready in {time.perf_counter() - t0:.2f}s (sample_rate={tts.sample_rate} Hz)")

    print("2) Synthesizing full-wave test line...")
    t1 = time.perf_counter()
    wav = tts.synthesize(args.text, seed=42)
    gen_time = time.perf_counter() - t1
    if wav.size == 0 or not np.all(np.isfinite(wav)):
        sys.exit("FAIL: Synthesized waveform is empty or non-finite.")
    duration = len(wav) / tts.sample_rate
    if duration < 0.4 or duration > 35.0:
        sys.exit(f"FAIL: Implausible duration {duration:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), wav, tts.sample_rate)
    rms = float(np.sqrt(np.mean(wav * wav)))
    print(
        f"   Saved {args.output} | duration={duration:.2f}s | gen={gen_time:.2f}s "
        f"| RTF={gen_time / max(duration, 1e-6):.2f} | RMS={rms:.4f}"
    )

    print("3) Testing streaming synthesis + early cancel recovery...")
    stop_ev = threading.Event()
    chunks = []
    for i, ch in enumerate(tts.stream_speech("Affirmative, Charles. All systems are online.", stop_event=stop_ev, seed=42)):
        chunks.append(ch)
        if i >= 2:
            stop_ev.set()
            break
    if not chunks:
        sys.exit("FAIL: Streaming synthesis produced no chunks.")
    recovery = tts.synthesize("Affirmative.", seed=42)
    if recovery.size == 0 or not np.all(np.isfinite(recovery)):
        sys.exit("FAIL: Post-cancel synthesis failed.")
    print(f"   Streaming cancel + recovery OK ({len(chunks)} partial chunks, recovery={len(recovery)/tts.sample_rate:.2f}s)")

    print("4) Round-trip verification with Whisper Large v3...")
    wav_16k = librosa.resample(wav, orig_sr=tts.sample_rate, target_sr=16000)
    stt = WhisperSTT()
    transcript = stt.transcribe(wav_16k, 16000)
    print(f"   Whisper heard: {transcript!r}")
    if not transcript:
        sys.exit("FAIL: Whisper returned an empty transcript.")
    expected_words = _norm_words(args.text)
    heard_words = _norm_words(transcript)
    overlap = len(expected_words & heard_words) / max(len(expected_words), 1)
    if overlap < 0.35:
        sys.exit(f"FAIL: Transcript overlap too low ({overlap:.2f}): {transcript!r}")

    print("5) Testing Liquid brain (Ollama)...")
    brain = OllamaBrain()
    try:
        reply_parts = list(
            brain.stream_reply_sentences("HAL, introduce yourself and confirm all systems are ready.")
        )
        reply = " ".join(reply_parts)
        if not reply:
            sys.exit("FAIL: Liquid returned an empty reply.")
        print(f"   Liquid ({brain.model}) replied: {reply!r}")
    finally:
        brain.close()

    model_out = args.output.with_name("hal9000_model_reply.wav")
    t2 = time.perf_counter()
    model_wav = tts.synthesize(reply, seed=42)
    model_dur = len(model_wav) / tts.sample_rate
    sf.write(str(model_out), model_wav, tts.sample_rate)
    print(
        f"   Saved {model_out} | duration={model_dur:.2f}s | gen={time.perf_counter() - t2:.2f}s"
    )

    if args.play:
        import sounddevice as sd

        print("6) Playing synthesized audio...")
        sd.play(wav, tts.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
