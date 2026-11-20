import { useEffect, useRef } from 'react';
import type { HalState } from './halClient';

export type { HalState };

export interface HalEyeProps {
  state: HalState;
  levelRef: React.MutableRefObject<number>;
  disabled: boolean;
  engaged: boolean;
  latched?: boolean;
  /** Normalized 0..1 capture progress toward max recording limit while listening. */
  captureProgressRef?: React.MutableRefObject<number>;
  onPointerStart: (pointerId: number) => void;
  onPointerEnd: (pointerId: number) => void;
  onPointerCancel: (pointerId: number) => void;
  onKeyboardToggle: () => void;
}

export function HalEye({
  state,
  levelRef,
  disabled,
  engaged,
  latched = false,
  captureProgressRef,
  onPointerStart,
  onPointerEnd,
  onPointerCancel,
  onKeyboardToggle,
}: HalEyeProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef<HalState>(state);
  const latchedRef = useRef<boolean>(latched);
  const activePointerIdRef = useRef<number | null>(null);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    latchedRef.current = latched;
  }, [latched]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let rafId = 0;
    let phase = 0;
    let smoothLevel = 0;
    let lastTs = performance.now();
    let cssW = canvas.clientWidth || 280;
    let cssH = canvas.clientHeight || 280;
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

    const syncCanvasSize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      cssW = Math.max(1, rect.width);
      cssH = Math.max(1, rect.height);
      const targetW = Math.max(1, Math.round(cssW * dpr));
      const targetH = Math.max(1, Math.round(cssH * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
    };

    syncCanvasSize();
    const ro = new ResizeObserver(syncCanvasSize);
    ro.observe(canvas);

    const draw = (now: number) => {
      if (document.hidden) {
        lastTs = now;
        rafId = requestAnimationFrame(draw);
        return;
      }

      const dt = Math.min(0.1, Math.max(0.001, (now - lastTs) / 1000));
      lastTs = now;

      const dpr = window.devicePixelRatio || 1;
      ctx.save();
      ctx.scale(dpr, dpr);
      const w = cssW;
      const h = cssH;
      ctx.clearRect(0, 0, w, h);

      if (!reducedMotion) {
        phase += dt * 2.7; // Frame-rate independent (~0.045/frame at 60 Hz)
      }
      const rawLevel = Math.max(0, Math.min(1, levelRef.current));
      const alpha = 1 - Math.exp(-dt * 15);
      smoothLevel += (rawLevel - smoothLevel) * alpha;

      const cur = stateRef.current;
      let intensity = 0.45 + 0.08 * Math.sin(phase * 0.85);
      if (cur === 'listening') {
        intensity = 0.88 + 0.16 * Math.sin(phase * 3.1) + smoothLevel * 0.42;
      } else if (cur === 'speaking') {
        intensity = 0.62 + smoothLevel * 0.68;
      } else if (cur === 'thinking') {
        intensity = 0.56 + 0.34 * (0.5 + 0.5 * Math.sin(phase * 2.8));
      } else if (cur === 'loading') {
        intensity = 0.25 + 0.2 * (0.5 + 0.5 * Math.sin(phase * 1.4));
      } else if (cur === 'error') {
        intensity = 0.32 + 0.14 * Math.sin(phase * 5.0);
      }
      intensity = Math.max(0.2, Math.min(1.35, intensity));

      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) * 0.39;

      // Ambient crimson halo
      const haloR = radius * (1.42 + smoothLevel * 0.18);
      const halo = ctx.createRadialGradient(cx, cy, radius * 0.3, cx, cy, haloR);
      const haloAlpha = Math.min(0.88, 0.22 + 0.46 * intensity);
      halo.addColorStop(0, `rgba(255, 36, 20, ${haloAlpha})`);
      halo.addColorStop(0.55, `rgba(180, 12, 12, ${haloAlpha * 0.45})`);
      halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
      ctx.fill();

      // Machined stainless outer bezel
      const bezel = ctx.createLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius);
      bezel.addColorStop(0.0, '#c8cfd8');
      bezel.addColorStop(0.22, '#4a4f58');
      bezel.addColorStop(0.48, '#a4adb8');
      bezel.addColorStop(0.76, '#2b2e35');
      bezel.addColorStop(1.0, '#8d96a2');
      ctx.fillStyle = bezel;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Inner bevel groove
      const grooveR = radius * 0.91;
      const groove = ctx.createLinearGradient(cx + grooveR, cy + grooveR, cx - grooveR, cy - grooveR);
      groove.addColorStop(0, '#5a606b');
      groove.addColorStop(0.5, '#111318');
      groove.addColorStop(1, '#30343c');
      ctx.fillStyle = groove;
      ctx.beginPath();
      ctx.arc(cx, cy, grooveR, 0, Math.PI * 2);
      ctx.fill();

      // Deep optical recess
      const recessR = radius * 0.85;
      ctx.fillStyle = '#070709';
      ctx.beginPath();
      ctx.arc(cx, cy, recessR, 0, Math.PI * 2);
      ctx.fill();

      // Fisheye lens glass
      const lensR = radius * 0.78;
      const lens = ctx.createRadialGradient(cx, cy, 0, cx, cy, lensR);
      lens.addColorStop(0.0, `rgba(255, 244, 178, ${Math.min(1, 0.76 * intensity)})`);
      lens.addColorStop(0.14, `rgba(255, 118, 22, ${Math.min(1, 0.92 * intensity)})`);
      lens.addColorStop(0.36, `rgba(224, 16, 12, ${Math.min(1, 0.94 * intensity)})`);
      lens.addColorStop(0.68, 'rgba(116, 4, 6, 0.96)');
      lens.addColorStop(0.9, 'rgba(26, 2, 4, 1)');
      lens.addColorStop(1.0, 'rgba(6, 1, 2, 1)');
      ctx.fillStyle = lens;
      ctx.beginPath();
      ctx.arc(cx, cy, lensR, 0, Math.PI * 2);
      ctx.fill();

      // Internal optics ring (gentle iris sweep while thinking)
      ctx.strokeStyle = `rgba(255, 90, 60, ${0.14 + 0.14 * intensity})`;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      if (cur === 'thinking' && !reducedMotion) {
        const sweep = phase * 1.9;
        ctx.arc(cx, cy, lensR * 0.56, sweep, sweep + Math.PI * 1.35);
      } else {
        ctx.arc(cx, cy, lensR * 0.56, 0, Math.PI * 2);
      }
      ctx.stroke();

      // Incandescent central pinpoint
      const coreR = radius * (0.095 + 0.045 * Math.min(1.1, intensity));
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
      core.addColorStop(0.0, 'rgba(255, 254, 228, 1)');
      core.addColorStop(0.5, 'rgba(255, 174, 48, 0.88)');
      core.addColorStop(1.0, 'rgba(255, 28, 12, 0)');
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
      ctx.fill();

      // Specular glass reflections
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.24)';
      ctx.lineWidth = Math.max(2.5, radius * 0.036);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(cx, cy, lensR * 0.74, Math.PI * 1.12, Math.PI * 1.38);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.10)';
      ctx.lineWidth = Math.max(1.8, radius * 0.02);
      ctx.beginPath();
      ctx.arc(cx, cy, lensR * 0.48, Math.PI * 1.15, Math.PI * 1.32);
      ctx.stroke();

      ctx.strokeStyle = `rgba(255, 110, 80, ${0.15 + 0.12 * intensity})`;
      ctx.lineWidth = Math.max(1.5, radius * 0.018);
      ctx.beginPath();
      ctx.arc(cx, cy, lensR * 0.72, Math.PI * 0.16, Math.PI * 0.34);
      ctx.stroke();

      // Active capture ring + latched reticle + countdown arc
      if (cur === 'listening') {
        const ringR = radius * (1.07 + smoothLevel * 0.06);
        const prog = Math.max(0, Math.min(1, captureProgressRef?.current ?? 0));
        const warn = prog > 0.82;

        if (latchedRef.current) {
          ctx.setLineDash([8, 6]);
        }
        ctx.strokeStyle = warn
          ? `rgba(255, 176, 48, ${0.7 + 0.25 * Math.sin(phase * 5)})`
          : `rgba(255, 86, 62, ${0.55 + 0.25 * Math.sin(phase * 4)})`;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (prog > 0.01) {
          ctx.strokeStyle = warn ? '#ffb030' : '#ff4733';
          ctx.lineWidth = 3.2;
          ctx.beginPath();
          ctx.arc(cx, cy, ringR + 5, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * prog);
          ctx.stroke();
        }
      }

      ctx.restore();
      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => {
      ro.disconnect();
      cancelAnimationFrame(rafId);
    };
  }, [captureProgressRef, levelRef]);

  return (
    <button
      type="button"
      data-ptt-control="true"
      className={`eye-stage ${disabled ? 'eye-disabled' : ''} eye-${state}`}
      disabled={disabled}
      aria-pressed={engaged}
      aria-label="Hold or click to speak to HAL 9000"
      onPointerDown={(e) => {
        if (disabled || e.button !== 0 || activePointerIdRef.current !== null) return;
        e.preventDefault();
        activePointerIdRef.current = e.pointerId;
        try {
          e.currentTarget.setPointerCapture(e.pointerId);
        } catch {
          // ignore if capture unavailable
        }
        onPointerStart(e.pointerId);
      }}
      onPointerUp={(e) => {
        if (activePointerIdRef.current !== e.pointerId) return;
        activePointerIdRef.current = null;
        onPointerEnd(e.pointerId);
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          // ignore
        }
      }}
      onPointerCancel={(e) => {
        if (activePointerIdRef.current !== e.pointerId) return;
        activePointerIdRef.current = null;
        onPointerCancel(e.pointerId);
      }}
      onLostPointerCapture={(e) => {
        if (activePointerIdRef.current !== e.pointerId) return;
        activePointerIdRef.current = null;
        onPointerCancel(e.pointerId);
      }}
      onClick={(e) => {
        // Keyboard Enter activation has detail === 0; pointer gestures are handled via PointerEvents
        if (e.detail !== 0) return;
        if (!disabled) onKeyboardToggle();
      }}
    >
      <canvas ref={canvasRef} className="eye-canvas" />
    </button>
  );
}
