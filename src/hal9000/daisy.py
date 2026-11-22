"""HAL 9000 singing renderer for *Daisy Bell (Bicycle Built for Two)*.

Synthesizes each lyric unit in HAL's own VoxCPM2 voice, pitch-locks voiced vowels to
Harry Dacre's 1892 waltz melody in HAL's baritone register via formant-preserving
TD-PSOLA (leaving unvoiced consonants crisp and natural), and applies an authentic
*2001: A Space Odyssey* analog varispeed power-down deceleration across the final lines.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .interfaces import TTSProvider

logger = logging.getLogger(__name__)

# Bb Major (HAL baritone register: F2=87.31 Hz .. F3=174.61 Hz, tonic Bb2=116.54 Hz)
DAISY_NOTES_HZ: dict[str, float] = {
    "F2": 87.31,
    "G2": 98.00,
    "A2": 110.00,
    "Bb2": 116.54,
    "C3": 130.81,
    "D3": 146.83,
    "Eb3": 155.56,
    "F3": 174.61,
}

# 3/4 waltz tempo (~143 BPM before the movie power-down varispeed stretches it out)
_BEAT_S = 0.42
_STREAM_CHUNK_S = 0.10
_CACHE_VERSION = "v1"

# Each phrase: (display_lyric, (rate_start, rate_end), [(spoken_text, [(note, beats), ...], pause_s), ...])
DAISY_SCORE: tuple[
    tuple[str, tuple[float, float], tuple[tuple[str, tuple[tuple[str, float], ...], float], ...]],
    ...,
] = (
    (
        "Daisy, Daisy, give me your answer, do.",
        (1.00, 0.96),
        (
            ("Daisy.", (("F3", 2.4), ("D3", 2.4)), 0.14),
            ("Daisy.", (("Bb2", 2.4), ("F2", 2.4)), 0.14),
            ("Give me your", (("G2", 1.0), ("A2", 1.0), ("Bb2", 1.0)), 0.03),
            ("answer,", (("G2", 1.9), ("Bb2", 1.0)), 0.04),
            ("do.", (("F2", 4.2),), 0.28),
        ),
    ),
    (
        "I'm half crazy, all for the love of you.",
        (0.96, 0.88),
        (
            ("I'm", (("C3", 2.3),), 0.04),
            ("half", (("F3", 2.3),), 0.04),
            ("crazy,", (("D3", 2.3), ("Bb2", 2.3)), 0.14),
            ("all for the", (("G2", 1.0), ("A2", 1.0), ("Bb2", 1.0)), 0.03),
            ("love", (("C3", 1.9),), 0.03),
            ("of", (("D3", 1.0),), 0.03),
            ("you.", (("C3", 4.2),), 0.30),
        ),
    ),
    (
        "It won't be a stylish marriage, I can't afford a carriage,",
        (0.88, 0.70),
        (
            ("It won't be a", (("D3", 0.95), ("Eb3", 0.95), ("D3", 0.95), ("C3", 0.95)), 0.03),
            ("stylish", (("F3", 1.0), ("D3", 1.0)), 0.03),
            ("marriage,", (("C3", 1.0), ("Bb2", 3.2)), 0.22),
            ("I can't", (("C3", 0.95), ("D3", 1.8)), 0.03),
            ("afford a", (("Bb2", 0.95), ("G2", 1.8), ("Bb2", 0.95)), 0.03),
            ("carriage,", (("G2", 1.0), ("F2", 3.2)), 0.26),
        ),
    ),
    (
        "but you'll look sweet upon the seat of a bicycle built for two.",
        (0.70, 0.44),
        (
            ("But you'll look", (("F2", 0.95), ("Bb2", 1.8), ("D3", 0.95)), 0.03),
            ("sweet", (("C3", 2.2),), 0.05),
            ("upon the", (("F2", 0.95), ("Bb2", 1.8), ("D3", 0.95)), 0.03),
            ("seat", (("C3", 1.6),), 0.06),
            ("of a", (("D3", 0.95), ("Eb3", 0.95)), 0.03),
            ("bicycle", (("F3", 1.05), ("D3", 1.05), ("Bb2", 1.05)), 0.05),
            ("built", (("C3", 2.0),), 0.06),
            ("for", (("F2", 1.3),), 0.08),
            ("two.", (("Bb2", 4.8),), 0.0),
        ),
    ),
)

DAISY_LYRIC_PHRASES: tuple[str, ...] = tuple(item[0] for item in DAISY_SCORE)
DAISY_PHRASE_INDEX: dict[str, int] = {text: idx for idx, text in enumerate(DAISY_LYRIC_PHRASES)}

_CACHE_LOCK = threading.Lock()
_PHRASE_MEM_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def _tts_cache_key(tts: TTSProvider) -> str:
    """Stable identifier for disk/memory caching keyed by reference audio and sample rate."""
    ref_wav = getattr(tts, "reference_wav", None)
    if ref_wav is None:
        return f"mem:{type(tts).__name__}:{id(tts)}"
    path = Path(str(ref_wav))
    try:
        st = path.stat()
        sig = f"{_CACHE_VERSION}:{path.resolve()}:{st.st_size}:{int(st.st_mtime)}:{tts.sample_rate}"
    except OSError:
        sig = f"{_CACHE_VERSION}:{path}:{tts.sample_rate}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def _disk_cache_path(cache_key: str, phrase_idx: int, sample_rate: int) -> Path | None:
    if cache_key.startswith("mem:"):
        return None
    cache_dir = Path.home() / ".cache" / "ioghal"
    return cache_dir / f"daisy_{_CACHE_VERSION}_{cache_key}_p{phrase_idx}_{sample_rate}.npy"


def _extract_pitch_marks(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(marks_samples, mark_f0_hz, mark_voiced)`` with consistent glottal polarity."""
    import librosa

    sr_low = 16000
    y_low = librosa.resample(y, orig_sr=sr, target_sr=sr_low) if sr != sr_low else y
    hop_low = 160
    f0_raw, voiced_flag, _ = librosa.pyin(
        y_low,
        fmin=60.0,
        fmax=260.0,
        sr=sr_low,
        frame_length=1024,
        hop_length=hop_low,
        fill_na=np.nan,
    )
    n_frames = len(f0_raw)
    if n_frames == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=bool),
        )
    valid_f0 = f0_raw[voiced_flag & np.isfinite(f0_raw)]
    med_f0 = float(np.median(valid_f0)) if valid_f0.size > 0 else 105.0
    f0_filled = np.where(voiced_flag & np.isfinite(f0_raw), f0_raw, med_f0)

    # Pick dominant glottal polarity across voiced speech so all grains phase-align cleanly
    if valid_f0.size > 0:
        p99 = float(np.percentile(y, 99.0))
        p01 = float(np.percentile(y, 1.0))
        sign = 1.0 if p99 >= -p01 else -1.0
    else:
        sign = 1.0
    y_pol = y * sign

    hop_s = hop_low / float(sr_low)

    def _query(sample_idx: int) -> tuple[float, bool]:
        fi = int(np.clip(round((sample_idx / float(sr)) / hop_s), 0, n_frames - 1))
        return float(f0_filled[fi]), bool(voiced_flag[fi] and np.isfinite(f0_raw[fi]))

    marks: list[int] = []
    f0s: list[float] = []
    voiceds: list[bool] = []
    n = len(y)
    pos = 0
    while pos < n:
        f0_cur, is_v = _query(pos)
        period = max(round(sr / max(f0_cur, 55.0)), 64)
        if is_v:
            lo = max(0, pos - period // 4)
            hi = min(n, pos + period // 4 + 1)
            peak = lo + int(np.argmax(y_pol[lo:hi])) if hi > lo else pos
            if marks and peak <= marks[-1] + period // 3:
                peak = min(n - 1, marks[-1] + period)
            marks.append(peak)
            f0s.append(f0_cur)
            voiceds.append(True)
            pos = peak + period
        else:
            marks.append(pos)
            f0s.append(med_f0)
            voiceds.append(False)
            pos += max(period // 2, 160)
    return (
        np.asarray(marks, dtype=np.int64),
        np.asarray(f0s, dtype=np.float32),
        np.asarray(voiceds, dtype=bool),
    )


def _split_syllables(y: np.ndarray, sr: int, n_syl: int) -> list[tuple[int, int]]:
    """Split a trimmed word/short phrase into ``n_syl`` spans at RMS energy valleys."""
    n = len(y)
    if n_syl <= 1 or n < int(0.08 * sr):
        return [(0, n)]
    import librosa

    hop = max(1, int(0.005 * sr))
    win = max(hop * 4, int(0.025 * sr))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    cuts = [0]
    min_seg = int(0.03 * sr)
    for k in range(1, n_syl):
        center = k / float(n_syl)
        lo_f = max(1, int((center - 0.16) * len(rms)))
        hi_f = min(len(rms) - 1, int((center + 0.16) * len(rms)))
        valley_f = lo_f + int(np.argmin(rms[lo_f:hi_f])) if hi_f > lo_f else round(center * len(rms))
        hi_bound = max(cuts[-1] + min_seg, n - (n_syl - k) * min_seg)
        cut_s = int(np.clip(valley_f * hop, cuts[-1] + min_seg, hi_bound))
        cuts.append(cut_s)
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(n_syl)]


def _crossfade_concat(segments: list[np.ndarray], sr: int, fade_ms: float = 10.0) -> np.ndarray:
    """Concatenate audio segments with short cosine crossfades to prevent boundary clicks."""
    non_empty = [seg.astype(np.float32, copy=False) for seg in segments if seg.size > 0]
    if not non_empty:
        return np.empty(0, dtype=np.float32)
    if len(non_empty) == 1:
        return non_empty[0]
    xf_max = max(1, round((fade_ms / 1000.0) * sr))
    out = non_empty[0]
    for nxt in non_empty[1:]:
        xf = min(xf_max, out.size // 3, nxt.size // 3)
        if xf >= 4:
            t = np.linspace(0.0, np.pi, xf, dtype=np.float32)
            fade_in = 0.5 * (1.0 - np.cos(t))
            fade_out = 1.0 - fade_in
            blend = out[-xf:] * fade_out + nxt[:xf] * fade_in
            out = np.concatenate((out[:-xf], blend, nxt[xf:]))
        else:
            out = np.concatenate((out, nxt))
    return out.astype(np.float32, copy=False)


def psola_sing_segment(
    y: np.ndarray,
    sr: int,
    target_hz: float,
    target_dur_s: float,
    *,
    prev_hz: float | None = None,
) -> np.ndarray:
    """Pitch-lock and sustain one syllable via formant-preserving TD-PSOLA."""
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size < int(0.025 * sr) or sr < 8000:
        return arr

    marks, mark_f0, mark_v = _extract_pitch_marks(arr, sr)
    voiced_idx = np.flatnonzero(mark_v)
    if voiced_idx.size < 3:
        return arr

    n_in = arr.size
    target_samples = max(round(target_dur_s * sr), int(0.06 * sr))
    v_start_s = int(marks[voiced_idx[0]])
    v_end_s = int(marks[voiced_idx[-1]])
    onset = arr[:v_start_s]
    coda = arr[v_end_s:]
    voiced_dur_target = max(target_samples - onset.size - coda.size, int(0.06 * sr))

    n_v = voiced_idx.size
    attack_n = max(1, round(0.24 * n_v))
    release_n = max(1, round(0.18 * n_v))
    mid_lo = min(attack_n, n_v - 1)
    mid_hi = max(mid_lo, n_v - release_n - 1)

    t_syn_base = max(round(sr / max(target_hz, 45.0)), 64)
    # Build synthesis mark positions with gentle 25ms portamento glide from previous note
    glide_samples = int(0.025 * sr)
    syn_positions: list[int] = []
    syn_periods: list[int] = []
    cur = 0
    while cur < voiced_dur_target:
        if prev_hz is not None and cur < glide_samples:
            alpha = cur / float(max(glide_samples, 1))
            hz_now = prev_hz * (1.0 - alpha) + target_hz * alpha
        else:
            hz_now = target_hz
        t_now = max(round(sr / max(hz_now, 45.0)), 64)
        syn_positions.append(cur)
        syn_periods.append(t_now)
        cur += t_now

    n_out_marks = len(syn_positions)
    pad = max(t_syn_base, round(sr / 55.0)) + 64
    out_len = voiced_dur_target + 2 * pad
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)

    for k in range(n_out_marks):
        u = k / float(max(n_out_marks - 1, 1))
        if u < 0.15:
            vi_f = (u / 0.15) * mid_lo
        elif u > 0.85:
            vi_f = mid_hi + ((u - 0.85) / 0.15) * (n_v - 1 - mid_hi)
        elif mid_hi > mid_lo:
            span = mid_hi - mid_lo
            cycles = max(1.0, (voiced_dur_target / float(max(v_end_s - v_start_s, 1))) * 0.65)
            phase = ((u - 0.15) / 0.70) * cycles
            tri = 1.0 - abs((phase % 2.0) - 1.0)
            vi_f = mid_lo + tri * span
        else:
            vi_f = float(mid_lo)

        v_idx = int(np.clip(round(vi_f), 0, n_v - 1))
        m_i = int(voiced_idx[v_idx])
        m_pos = int(marks[m_i])
        t_ana = max(round(sr / max(float(mark_f0[m_i]), 55.0)), 64)
        # Ensure window covers synthesis period so downward shifts never leave zero-gaps
        half_w = max(t_ana, round(syn_periods[k] * 1.05))

        win_len = 2 * half_w + 1
        win = np.hanning(win_len)
        src_lo = m_pos - half_w
        src_hi = m_pos + half_w + 1
        grain = np.zeros(win_len, dtype=np.float64)
        g_lo = max(0, -src_lo)
        g_hi = win_len - max(0, src_hi - n_in)
        s_lo = max(0, src_lo)
        s_hi = min(n_in, src_hi)
        if s_hi > s_lo:
            grain[g_lo:g_hi] = arr[s_lo:s_hi]
        grain *= win

        syn_center = pad + syn_positions[k]
        d_lo = syn_center - half_w
        d_hi = syn_center + half_w + 1
        if d_lo < 0:
            grain = grain[-d_lo:]
            win = win[-d_lo:]
            d_lo = 0
        if d_hi > out_len:
            trim = d_hi - out_len
            grain = grain[:-trim]
            win = win[:-trim]
            d_hi = out_len
        if d_hi > d_lo:
            out[d_lo:d_hi] += grain
            wsum[d_lo:d_hi] += win

    valid = wsum > 1e-4
    out[valid] /= wsum[valid]
    voiced_out = out[pad : pad + voiced_dur_target].astype(np.float32)

    # Gentle natural breath decay on sustained notes (>0.6s)
    if target_dur_s > 0.6 and voiced_out.size > 16:
        voiced_out *= np.linspace(1.0, 0.88, voiced_out.size, dtype=np.float32)

    return _crossfade_concat([onset, voiced_out, coda], sr, fade_ms=8.0)


def apply_varispeed(y: np.ndarray, rate_start: float, rate_end: float) -> np.ndarray:
    """True analog tape power-down varispeed (smoothly slows tempo and lowers pitch)."""
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    n_in = arr.size
    if n_in < 4 or (abs(rate_start - 1.0) < 1e-4 and abs(rate_end - 1.0) < 1e-4):
        return arr
    r0 = max(float(rate_start), 0.15)
    r1 = max(float(rate_end), 0.15)
    if abs(r1 - r0) < 1e-4:
        n_out = max(4, round(n_in / r0))
        src_pos = np.linspace(0.0, n_in - 1.0, n_out, dtype=np.float64)
    else:
        n_out = max(4, round(n_in * np.log(r1 / r0) / (r1 - r0)))
        k = (r1 - r0) / float(n_in)
        t = np.arange(n_out, dtype=np.float64)
        src_pos = np.clip((r0 / k) * (np.exp(k * t) - 1.0), 0.0, n_in - 1.0)
    cs = CubicSpline(np.arange(n_in, dtype=np.float64), arr.astype(np.float64), bc_type="natural")
    return cs(src_pos).astype(np.float32)


def _render_daisy_phrase(
    tts: TTSProvider,
    phrase_idx: int,
    *,
    stop_event: threading.Event | None = None,
    inference_timesteps: int | None = None,
) -> np.ndarray | None:
    """Render one sung phrase of *Daisy Bell* in HAL's voice with movie varispeed."""
    import librosa

    _, (rate_start, rate_end), units = DAISY_SCORE[phrase_idx]
    sr = int(getattr(tts, "sample_rate", 48000))
    phrase_segments: list[np.ndarray] = []
    prev_hz: float | None = None

    for text, syl_notes, pause_s in units:
        if stop_event is not None and stop_event.is_set():
            return None
        tts_kwargs: dict = {"stop_event": stop_event}
        if inference_timesteps is not None:
            tts_kwargs["inference_timesteps"] = inference_timesteps
        raw_parts = [
            np.asarray(ch, dtype=np.float32).reshape(-1)
            for ch in tts.stream_speech(text, **tts_kwargs)
            if np.asarray(ch).size > 0
        ]
        if stop_event is not None and stop_event.is_set():
            return None
        if not raw_parts:
            continue
        raw = np.concatenate(raw_parts)
        trimmed, _ = librosa.effects.trim(raw, top_db=28)
        if trimmed.size < int(0.04 * sr):
            trimmed = raw
        spans = _split_syllables(trimmed, sr, len(syl_notes))
        unit_parts: list[np.ndarray] = []
        for (s0, s1), (note_name, beats) in zip(spans, syl_notes, strict=True):
            if stop_event is not None and stop_event.is_set():
                return None
            note_hz = DAISY_NOTES_HZ[note_name]
            sung = psola_sing_segment(
                trimmed[s0:s1],
                sr,
                note_hz,
                beats * _BEAT_S,
                prev_hz=prev_hz,
            )
            unit_parts.append(sung)
            prev_hz = note_hz
        phrase_segments.append(_crossfade_concat(unit_parts, sr, fade_ms=10.0))
        if pause_s > 0:
            phrase_segments.append(np.zeros(round(pause_s * sr), dtype=np.float32))
            prev_hz = None

    if not phrase_segments:
        return np.empty(0, dtype=np.float32)

    phrase_wav = _crossfade_concat(phrase_segments, sr, fade_ms=6.0)
    phrase_wav = apply_varispeed(phrase_wav, rate_start, rate_end)

    # On the final line ("...built for two."), add the movie's haunting power-down fade-out
    if phrase_idx == len(DAISY_SCORE) - 1 and phrase_wav.size > 32:
        fade_n = min(int(1.4 * sr), phrase_wav.size // 3)
        phrase_wav[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32) ** 1.6

    # Match HAL's calm spoken RMS/peak level (~0.70 peak) cleanly without clipping
    peak = float(np.max(np.abs(phrase_wav)))
    if peak > 1e-4:
        phrase_wav = (phrase_wav * (0.70 / peak)).astype(np.float32)
    return np.ascontiguousarray(phrase_wav, dtype="<f4")


def stream_daisy_phrase(
    tts: TTSProvider,
    phrase_idx: int,
    *,
    stop_event: threading.Event | None = None,
    inference_timesteps: int | None = None,
) -> Iterator[np.ndarray]:
    """Stream one sung phrase of *Daisy Bell* in ~100ms chunks with caching and instant cancel."""
    sr = int(getattr(tts, "sample_rate", 48000))
    key_id = _tts_cache_key(tts)
    mem_key = (key_id, phrase_idx, sr)

    wav: np.ndarray | None = None
    with _CACHE_LOCK:
        wav = _PHRASE_MEM_CACHE.get(mem_key)

    if wav is None:
        disk_path = _disk_cache_path(key_id, phrase_idx, sr)
        if disk_path is not None and disk_path.exists():
            try:
                loaded = np.load(disk_path)
                if isinstance(loaded, np.ndarray) and loaded.ndim == 1 and loaded.size > 0:
                    wav = np.ascontiguousarray(loaded, dtype="<f4")
                    with _CACHE_LOCK:
                        _PHRASE_MEM_CACHE[mem_key] = wav
            except (OSError, ValueError):
                logger.debug("Ignoring unreadable Daisy cache %s", disk_path)

    if wav is None:
        wav = _render_daisy_phrase(
            tts,
            phrase_idx,
            stop_event=stop_event,
            inference_timesteps=inference_timesteps,
        )
        if wav is None or wav.size == 0:
            return
        with _CACHE_LOCK:
            _PHRASE_MEM_CACHE[mem_key] = wav
        disk_path = _disk_cache_path(key_id, phrase_idx, sr)
        if disk_path is not None:
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = disk_path.with_suffix(".tmp.npy")
                np.save(tmp_path, wav)
                tmp_path.replace(disk_path)
            except OSError:
                logger.debug("Could not persist Daisy cache %s", disk_path)

    chunk_samples = max(1, round(_STREAM_CHUNK_S * sr))
    for offset in range(0, wav.size, chunk_samples):
        if stop_event is not None and stop_event.is_set():
            return
        yield wav[offset : offset + chunk_samples]


def clear_daisy_cache() -> None:
    """Clear in-memory cached Daisy phrases (useful in tests)."""
    with _CACHE_LOCK:
        _PHRASE_MEM_CACHE.clear()
