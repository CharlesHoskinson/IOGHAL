"""HAL 9000 Push-to-Talk Desktop & CLI Application (VoxCPM2 + Qwen + Whisper)."""

from __future__ import annotations

import argparse
import html
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from .audio import MicRecorder, StreamingAudioPlayer
from .brain import QwenBrain, choose_default_qwen_model, list_ollama_models
from .stt import WhisperSTT
from .tts import MIN_DIFFUSION_STEPS, HalVoiceEngine, resolve_reference_wav

logger = logging.getLogger(__name__)


def run_cli_mode(
    model: str | None,
    ref_wav: Path | None,
    ollama_url: str,
    compile_model: bool,
    steps: int,
) -> None:
    """Minimal terminal Push-to-Talk (Enter to start/stop) fallback."""
    print("[IOGHAL] Initializing neural subsystems...")
    brain = QwenBrain(model=model, base_url=ollama_url)
    stt = WhisperSTT()
    tts = HalVoiceEngine(
        reference_wav=resolve_reference_wav(ref_wav),
        optimize=compile_model,
        inference_timesteps=steps,
    )
    mic = MicRecorder()
    player = StreamingAudioPlayer(sample_rate=tts.sample_rate)

    print(f"[IOGHAL] Online. Brain={brain.model}, Voice=VoxCPM2 ({tts.sample_rate} Hz).")
    print("Press ENTER to start speaking, ENTER again to stop (Ctrl+C to exit).\n")

    try:
        while True:
            input("Press ENTER to talk...")
            player.stop()
            try:
                mic.start()
            except (sd.PortAudioError, OSError, RuntimeError) as exc:
                print(f"[ERROR] Microphone start failed: {exc}\n")
                continue
            print("● Listening... (press ENTER to send)")
            input()
            audio = mic.stop()

            try:
                t0 = time.perf_counter()
                user_text = stt.transcribe(audio, mic.sample_rate)
                t_stt = time.perf_counter() - t0
                if not user_text:
                    print("(No speech detected)\n")
                    continue

                print(f"You ({t_stt:.2f}s): {user_text}")
                print("HAL: ", end="", flush=True)
                stop_ev = threading.Event()
                player.start_session()
                for sentence in brain.stream_reply_sentences(user_text):
                    print(sentence + " ", end="", flush=True)
                    for chunk in tts.stream_speech(sentence, stop_event=stop_ev):
                        player.enqueue(chunk)
                print("\n")
                player.finish()
                if player.last_error is not None:
                    print(f"[ERROR] Audio playback failed: {player.last_error}\n")
            except Exception as exc:
                logger.exception("CLI turn failed")
                player.stop()
                print(f"\n[ERROR] Turn failed: {exc}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n[IOGHAL] Shutting down. Daisy, Daisy...")
    finally:
        mic.stop()
        player.stop()
        brain.close()


def run_gui_mode(
    model: str | None,
    ref_wav: Path,
    ollama_url: str,
    compile_model: bool,
    steps: int,
) -> None:
    """Launch the PyQt6 HAL 9000 Optical Eye Push-to-Talk desktop application."""
    from PyQt6.QtCore import QEvent, QObject, QPointF, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QColor,
        QKeyEvent,
        QLinearGradient,
        QPainter,
        QPen,
        QRadialGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class Signals(QObject):
        ready = pyqtSignal(str, int)
        init_error = pyqtSignal(str)
        status = pyqtSignal(int, str, str)  # (turn_id, state, label)
        level = pyqtSignal(float)
        append_log = pyqtSignal(int, str, str)  # (turn_id, speaker, text)
        metrics = pyqtSignal(int, str)

    class HalEyeWidget(QWidget):
        """Interactive glowing HAL 9000 optical lens that doubles as a mouse PTT button."""

        ptt_pressed = pyqtSignal()
        ptt_released = pyqtSignal()

        def __init__(self) -> None:
            super().__init__()
            self.setMinimumSize(250, 250)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.state = "loading"
            self.audio_level = 0.0
            self.smooth_level = 0.0
            self.phase = 0.0
            self._mouse_down = False

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(16)

        def set_state(self, state: str) -> None:
            self.state = state
            self.update()

        def set_level(self, lvl: float) -> None:
            self.audio_level = max(0.0, min(1.0, lvl))

        def _tick(self) -> None:
            self.phase += 0.055
            self.smooth_level += (self.audio_level - self.smooth_level) * 0.25
            self.update()

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._mouse_down = True
                self.ptt_pressed.emit()
                event.accept()

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self._mouse_down:
                self._mouse_down = False
                self.ptt_released.emit()
                event.accept()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w, h = self.width(), self.height()
            cx, cy = w / 2.0, h / 2.0
            radius = min(w, h) * 0.43

            if self.state == "listening":
                intensity = 0.82 + 0.18 * math.sin(self.phase * 3.0) + self.smooth_level * 0.35
            elif self.state == "speaking":
                intensity = 0.65 + self.smooth_level * 0.55
            elif self.state == "thinking":
                intensity = 0.55 + 0.30 * (0.5 + 0.5 * math.sin(self.phase * 2.6))
            elif self.state == "loading":
                intensity = 0.30 + 0.20 * (0.5 + 0.5 * math.sin(self.phase * 1.5))
            else:
                intensity = 0.42 + 0.08 * math.sin(self.phase * 0.8)
            intensity = max(0.2, min(1.25, intensity))

            halo_r = radius * (1.28 + 0.14 * self.smooth_level)
            halo = QRadialGradient(QPointF(cx, cy), halo_r)
            halo_alpha = int(min(220, 70 + 115 * intensity))
            if self.state == "listening":
                halo.setColorAt(0.0, QColor(255, 65, 35, halo_alpha))
                halo.setColorAt(0.65, QColor(220, 20, 20, int(halo_alpha * 0.45)))
            else:
                halo.setColorAt(0.0, QColor(235, 25, 20, halo_alpha))
                halo.setColorAt(0.65, QColor(150, 10, 10, int(halo_alpha * 0.35)))
            halo.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QPointF(cx, cy), halo_r, halo_r)

            bezel = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
            bezel.setColorAt(0.0, QColor(185, 192, 200))
            bezel.setColorAt(0.25, QColor(72, 76, 82))
            bezel.setColorAt(0.55, QColor(150, 156, 165))
            bezel.setColorAt(0.85, QColor(42, 45, 50))
            bezel.setColorAt(1.0, QColor(125, 130, 138))
            painter.setBrush(bezel)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            inner_r = radius * 0.87
            painter.setBrush(QColor(12, 12, 15))
            painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

            lens_r = radius * 0.80
            lens = QRadialGradient(QPointF(cx, cy), lens_r)
            lens.setColorAt(0.0, QColor(255, 240, 160, int(min(255, 175 * intensity))))
            lens.setColorAt(0.14, QColor(255, 110, 25, int(min(255, 215 * intensity))))
            lens.setColorAt(0.38, QColor(225, 18, 12, int(min(255, 225 * intensity))))
            lens.setColorAt(0.72, QColor(115, 6, 6, 240))
            lens.setColorAt(0.92, QColor(32, 3, 4, 255))
            lens.setColorAt(1.0, QColor(8, 2, 3, 255))
            painter.setBrush(lens)
            painter.drawEllipse(QPointF(cx, cy), lens_r, lens_r)

            core_r = radius * (0.095 + 0.045 * min(1.0, intensity))
            core = QRadialGradient(QPointF(cx, cy), core_r)
            core.setColorAt(0.0, QColor(255, 252, 210, 255))
            core.setColorAt(0.55, QColor(255, 165, 50, 210))
            core.setColorAt(1.0, QColor(255, 30, 10, 0))
            painter.setBrush(core)
            painter.drawEllipse(QPointF(cx, cy), core_r, core_r)

            refl_pen = QPen(QColor(255, 255, 255, 42), max(2.0, radius * 0.035))
            refl_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(refl_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            arc_r = lens_r * 0.76
            painter.drawArc(
                int(cx - arc_r),
                int(cy - arc_r),
                int(arc_r * 2),
                int(arc_r * 2),
                110 * 16,
                50 * 16,
            )

            if self.state == "listening":
                ring_pen = QPen(QColor(255, 80, 60, 220), 3.0)
                painter.setPen(ring_pen)
                ring_r = radius * (1.06 + 0.06 * self.smooth_level)
                painter.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

    class HalMainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("HAL 9000 — VoxCPM2 + Qwen PTT")
            self.resize(520, 740)
            self.setMinimumSize(440, 620)
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background-color: #0b0c10;
                    color: #e6e8ec;
                    font-family: 'Inter', 'DejaVu Sans', sans-serif;
                }
                QLabel#title {
                    font-size: 16px;
                    font-weight: 700;
                    letter-spacing: 4px;
                    color: #f0f2f5;
                }
                QLabel#badge {
                    background-color: #161922;
                    border: 1px solid #2b3040;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-size: 12px;
                    font-weight: 600;
                    color: #ff6b57;
                }
                QLabel#metrics {
                    font-size: 11px;
                    color: #7d8596;
                    font-family: monospace;
                }
                QComboBox, QLineEdit {
                    background-color: #141720;
                    border: 1px solid #2a3042;
                    border-radius: 6px;
                    padding: 6px 10px;
                    color: #e6e8ec;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #1a1e2a;
                    border: 1px solid #30374c;
                    border-radius: 6px;
                    padding: 7px 14px;
                    color: #e6e8ec;
                    font-weight: 600;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #252b3d;
                    border-color: #46506e;
                }
                QPushButton#pttBtn {
                    background-color: #2a1114;
                    border: 1px solid #8a2020;
                    color: #ff8a7a;
                    padding: 12px 18px;
                    font-size: 13px;
                    letter-spacing: 1px;
                }
                QPushButton#pttBtn:pressed {
                    background-color: #6b1518;
                    border-color: #ff4433;
                    color: #ffffff;
                }
                QTextEdit {
                    background-color: #10121a;
                    border: 1px solid #232838;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 13px;
                    line-height: 1.45;
                    color: #dfe3eb;
                }
                """
            )

            self.signals = Signals()
            self.signals.ready.connect(self._on_ready)
            self.signals.init_error.connect(self._on_init_error)
            self.signals.status.connect(self._on_status)
            self.signals.level.connect(self._on_level)
            self.signals.append_log.connect(self._on_append_log)
            self.signals.metrics.connect(self._on_metrics)

            discovered_models = list_ollama_models(ollama_url)
            chosen_model = model or choose_default_qwen_model(ollama_url, models=discovered_models)
            self.brain = QwenBrain(model=chosen_model, base_url=ollama_url)
            self.stt: WhisperSTT | None = None
            self.tts: HalVoiceEngine | None = None
            self.mic = MicRecorder(
                on_level=self.signals.level.emit,
                on_max_duration=lambda: self.signals.status.emit(
                    0, "listening", "MAX DURATION REACHED — RELEASE TO SEND"
                ),
            )
            self.player: StreamingAudioPlayer | None = None
            self._is_ready = False
            self._capture_owner: str | None = None
            self._turn_lock = threading.Lock()
            self._state_lock = threading.Lock()
            self._active_turn_id = 0
            self._active_cancel = threading.Event()

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(12)

            top = QHBoxLayout()
            title = QLabel("IOGHAL")
            title.setObjectName("title")
            top.addWidget(title)
            top.addStretch()

            self.model_combo = QComboBox()
            self.model_combo.setMinimumWidth(190)
            available_models = discovered_models or [self.brain.model]
            if self.brain.model not in available_models:
                available_models = [self.brain.model, *available_models]
            self.model_combo.addItems(available_models)
            idx = self.model_combo.findText(self.brain.model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            self.model_combo.currentTextChanged.connect(self.brain.set_model)
            top.addWidget(self.model_combo)

            clear_btn = QPushButton("Reset")
            clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            clear_btn.clicked.connect(self._on_reset)
            top.addWidget(clear_btn)
            layout.addLayout(top)

            self.eye = HalEyeWidget()
            self.eye.ptt_pressed.connect(lambda: self._start_ptt("eye"))
            self.eye.ptt_released.connect(lambda: self._stop_ptt("eye"))
            layout.addWidget(self.eye, stretch=3)

            status_row = QHBoxLayout()
            self.status_badge = QLabel("INITIALIZING MODELS...")
            self.status_badge.setObjectName("badge")
            status_row.addWidget(self.status_badge)
            status_row.addStretch()
            self.metrics_label = QLabel("")
            self.metrics_label.setObjectName("metrics")
            status_row.addWidget(self.metrics_label)
            layout.addLayout(status_row)

            self.ptt_btn = QPushButton("HOLD [SPACE] OR PRESS & HOLD TO TALK")
            self.ptt_btn.setObjectName("pttBtn")
            self.ptt_btn.setEnabled(False)
            self.ptt_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.ptt_btn.pressed.connect(lambda: self._start_ptt("button"))
            self.ptt_btn.released.connect(lambda: self._stop_ptt("button"))
            layout.addWidget(self.ptt_btn)

            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            layout.addWidget(self.log, stretch=2)

            input_row = QHBoxLayout()
            self.text_input = QLineEdit()
            self.text_input.setPlaceholderText("Or type a message to HAL and press Enter...")
            self.text_input.setEnabled(False)
            self.text_input.returnPressed.connect(self._on_text_submit)
            input_row.addWidget(self.text_input)

            send_btn = QPushButton("Send")
            send_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            send_btn.clicked.connect(self._on_text_submit)
            self.send_btn = send_btn
            self.send_btn.setEnabled(False)
            input_row.addWidget(send_btn)
            layout.addLayout(input_row)

            QApplication.instance().installEventFilter(self)
            threading.Thread(target=self._init_models, daemon=True).start()

        def _next_turn(self) -> tuple[int, threading.Event]:
            with self._state_lock:
                self._active_cancel.set()
                self._active_turn_id += 1
                self._active_cancel = threading.Event()
                return self._active_turn_id, self._active_cancel

        def _retire_turn(self) -> None:
            with self._state_lock:
                self._active_cancel.set()
                self._active_turn_id += 1

        def _is_current(self, turn_id: int) -> bool:
            with self._state_lock:
                return turn_id == 0 or turn_id == self._active_turn_id

        def _init_models(self) -> None:
            try:
                self.signals.status.emit(0, "loading", "LOADING WHISPER LARGE-V3 (CUDA)...")
                self.stt = WhisperSTT()
                self.signals.status.emit(0, "loading", "LOADING VOXCPM2 + HAL 9000 VOICE...")
                self.tts = HalVoiceEngine(
                    reference_wav=ref_wav,
                    optimize=compile_model,
                    inference_timesteps=steps,
                )
                self.player = StreamingAudioPlayer(
                    sample_rate=self.tts.sample_rate,
                    on_level=self.signals.level.emit,
                )
                self.signals.ready.emit(self.brain.model, self.tts.sample_rate)
            except Exception as exc:
                logger.exception("Initialization failed")
                self.signals.init_error.emit(str(exc))

        def _on_ready(self, brain_model: str, sr: int) -> None:
            self._is_ready = True
            self.ptt_btn.setEnabled(True)
            self.text_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._on_status(0, "idle", "READY — HOLD SPACE OR EYE TO TALK")
            self._on_append_log(
                0,
                "SYSTEM",
                f"All circuits operational. Brain: {brain_model} | Voice: VoxCPM2 ({sr} Hz) | STT: Whisper Large v3.",
            )

        def _on_init_error(self, message: str) -> None:
            self._on_status(0, "idle", "INITIALIZATION ERROR")
            self._on_append_log(0, "ERROR", message)

        def _on_status(self, turn_id: int, state: str, text: str) -> None:
            if not self._is_current(turn_id):
                return
            self.eye.set_state(state)
            self.status_badge.setText(text)

        def _on_level(self, lvl: float) -> None:
            self.eye.set_level(lvl)

        def _on_metrics(self, turn_id: int, text: str) -> None:
            if not self._is_current(turn_id):
                return
            self.metrics_label.setText(text)

        def _on_append_log(self, turn_id: int, speaker: str, text: str) -> None:
            if not self._is_current(turn_id):
                return
            color = {
                "YOU": "#78b7ff",
                "HAL": "#ff6b57",
                "SYSTEM": "#8590a6",
                "ERROR": "#ff4444",
            }.get(speaker, "#cccccc")
            safe_speaker = html.escape(speaker)
            safe_text = html.escape(text)
            self.log.append(
                f'<span style="color:{color}; font-weight:700;">{safe_speaker}:</span> {safe_text}'
            )

        def _on_reset(self) -> None:
            self._cancel_all()
            self.brain.reset()
            self.log.clear()
            self._on_append_log(0, "SYSTEM", "Conversation memory cleared.")

        def _cancel_all(self) -> None:
            self._retire_turn()
            self._capture_owner = None
            if self.mic.is_recording:
                self.mic.stop()
                self.ptt_btn.setText("HOLD [SPACE] OR PRESS & HOLD TO TALK")
            if self.player is not None:
                self.player.stop()
            self._on_status(0, "idle", "READY — HOLD SPACE OR EYE TO TALK")

        def eventFilter(self, obj, event) -> bool:
            if not self._is_ready:
                return super().eventFilter(obj, event)
            if event.type() == QEvent.Type.ApplicationDeactivate:
                if self._capture_owner is not None or self.mic.is_recording:
                    self._cancel_all()
                return super().eventFilter(obj, event)
            if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_all()
                    return True
                if (
                    event.key() == Qt.Key.Key_Space
                    and not event.isAutoRepeat()
                    and not self.text_input.hasFocus()
                    and not self.model_combo.hasFocus()
                ):
                    if self._capture_owner is None:
                        self._start_ptt("space")
                    return True
            if (
                event.type() == QEvent.Type.KeyRelease
                and isinstance(event, QKeyEvent)
                and event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._capture_owner == "space"
            ):
                self._stop_ptt("space")
                return True
            return super().eventFilter(obj, event)

        def _start_ptt(self, owner: str = "space") -> None:
            if not self._is_ready or self.mic.is_recording or self._capture_owner is not None:
                return
            self._retire_turn()
            if self.player is not None:
                self.player.stop()
            try:
                self.mic.start()
            except (sd.PortAudioError, OSError, RuntimeError) as exc:
                self._capture_owner = None
                self._on_append_log(0, "ERROR", f"Microphone error: {exc}")
                self._on_status(0, "idle", "MICROPHONE ERROR")
                return
            self._capture_owner = owner
            self.ptt_btn.setText("● TRANSMITTING — RELEASE TO SEND")
            self._on_status(0, "listening", "LISTENING...")

        def _stop_ptt(self, owner: str = "space") -> None:
            if self._capture_owner != owner or not self.mic.is_recording:
                return
            self._capture_owner = None
            audio = self.mic.stop()
            self.ptt_btn.setText("HOLD [SPACE] OR PRESS & HOLD TO TALK")
            if audio.size < int(self.mic.sample_rate * 0.2):
                self._on_status(0, "idle", "READY — HOLD SPACE OR EYE TO TALK")
                return
            turn_id, turn_cancel = self._next_turn()
            threading.Thread(
                target=self._handle_turn,
                kwargs={"turn_id": turn_id, "turn_cancel": turn_cancel, "audio": audio},
                daemon=True,
            ).start()

        def _on_text_submit(self) -> None:
            if not self._is_ready or self.mic.is_recording:
                return
            text = self.text_input.text().strip()
            if not text:
                return
            self.text_input.clear()
            self.text_input.clearFocus()
            turn_id, turn_cancel = self._next_turn()
            if self.player is not None:
                self.player.stop()
            threading.Thread(
                target=self._handle_turn,
                kwargs={"turn_id": turn_id, "turn_cancel": turn_cancel, "user_text": text},
                daemon=True,
            ).start()

        def _handle_turn(
            self,
            turn_id: int,
            turn_cancel: threading.Event,
            audio: np.ndarray | None = None,
            user_text: str | None = None,
        ) -> None:
            with self._turn_lock:
                if turn_cancel.is_set() or not self._is_current(turn_id):
                    return
                assert self.stt is not None and self.tts is not None and self.player is not None

                t_stt = 0.0
                spoken_sentences: list[str] = []
                try:
                    if user_text is None:
                        assert audio is not None
                        self.signals.status.emit(turn_id, "thinking", "TRANSCRIBING...")
                        t0 = time.perf_counter()
                        user_text = self.stt.transcribe(audio, self.mic.sample_rate)
                        t_stt = time.perf_counter() - t0
                        if not user_text or turn_cancel.is_set():
                            self.signals.status.emit(
                                turn_id, "idle", "READY — HOLD SPACE OR EYE TO TALK"
                            )
                            return

                    self.signals.append_log.emit(turn_id, "YOU", user_text)
                    self.signals.status.emit(
                        turn_id, "thinking", f"THINKING ({self.brain.model})..."
                    )

                    t_llm_start = time.perf_counter()
                    t_first_audio: float | None = None
                    self.player.start_session()

                    for sentence in self.brain.stream_reply_sentences(
                        user_text, stop_check=turn_cancel.is_set
                    ):
                        if turn_cancel.is_set():
                            break
                        sentence_emitted = False
                        for chunk in self.tts.stream_speech(sentence, stop_event=turn_cancel):
                            if turn_cancel.is_set():
                                break
                            if not sentence_emitted:
                                sentence_emitted = True
                                spoken_sentences.append(sentence)
                            if t_first_audio is None:
                                t_first_audio = time.perf_counter() - t_llm_start
                                self.signals.status.emit(
                                    turn_id, "speaking", "HAL 9000 SPEAKING..."
                                )
                                self.signals.metrics.emit(
                                    turn_id,
                                    f"STT {t_stt:.2f}s | First chunk {t_first_audio:.2f}s",
                                )
                            self.player.enqueue(chunk)
                except Exception as exc:
                    logger.exception("Turn %d failed", turn_id)
                    self.signals.append_log.emit(turn_id, "ERROR", str(exc))
                    self.player.stop()
                    self.signals.status.emit(
                        turn_id, "idle", "READY — HOLD SPACE OR EYE TO TALK"
                    )
                    return

                if spoken_sentences and not turn_cancel.is_set():
                    self.signals.append_log.emit(turn_id, "HAL", " ".join(spoken_sentences))

                if turn_cancel.is_set():
                    self.player.stop()
                else:
                    self.player.finish()
                    if self.player.last_error is not None:
                        self.signals.append_log.emit(
                            turn_id, "ERROR", f"Playback failed: {self.player.last_error}"
                        )

                if not self.mic.is_recording:
                    self.signals.status.emit(
                        turn_id, "idle", "READY — HOLD SPACE OR EYE TO TALK"
                    )

        def closeEvent(self, event) -> None:
            self._cancel_all()
            self.eye._timer.stop()
            app_inst = QApplication.instance()
            if app_inst is not None:
                app_inst.removeEventFilter(self)
            self.brain.close()
            super().closeEvent(event)

    app = QApplication(sys.argv)
    app.setApplicationName("IOGHAL")
    window = HalMainWindow()
    window.show()
    sys.exit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IOGHAL Push-to-Talk Desktop & CLI Voice Assistant (VoxCPM2 + Qwen + Whisper)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama Qwen model name (default: auto-selects qwen3.5:9b-q8_0)",
    )
    parser.add_argument(
        "--ref-wav",
        type=Path,
        default=None,
        help="Path to voice reference WAV clip (default: voices/hal9000_ref.wav or HAL_REF_WAV)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in terminal Push-to-Talk mode instead of the PyQt6 GUI",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile optimization in VoxCPM2 (longer startup)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Diffusion inference timesteps for VoxCPM2 (default: 6; >= 2)",
    )
    args = parser.parse_args()

    if args.steps < MIN_DIFFUSION_STEPS:
        parser.error(f"--steps must be >= {MIN_DIFFUSION_STEPS}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ref_wav = resolve_reference_wav(args.ref_wav)
    if args.cli:
        run_cli_mode(args.model, ref_wav, args.ollama_url, args.compile, args.steps)
    else:
        run_gui_mode(args.model, ref_wav, args.ollama_url, args.compile, args.steps)


if __name__ == "__main__":
    main()
