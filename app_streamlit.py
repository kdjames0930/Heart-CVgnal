"""
Heart CV-gnal — Streamlit Cloud Version
========================================
브라우저 카메라 → streamlit-webrtc → 서버(Streamlit Cloud)에서 MediaPipe 처리.

로컬 실행:
    $env:PYTHONPATH="src"; streamlit run app_streamlit.py

Streamlit Cloud:
    main module: app_streamlit.py
    Python: 3.11  (via .python-version)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from heart_cvgnal.pipelines.vision.affection_engine import AffectionEngine
from heart_cvgnal.pipelines.vision.feature_extractor import FeatureExtractor

try:
    from heart_cvgnal.pipelines.vision.vlm_analyzer import VLMAnalyzer
    _HAS_VLM = True
except Exception:
    _HAS_VLM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEMO_DURATION = 180.0
_EVAL_WINDOW   = 5.0

_MOOD_LABELS = [
    (0,  "Not Feeling It"),
    (20, "Mildly Curious"),
    (40, "Warming Up"),
    (60, "Interested"),
    (75, "Smitten"),
    (88, "Head Over Heels"),
]

# BGR overlays
_HOT   = (147,  20, 255)
_MED   = (180, 105, 255)
_LIGHT = (203, 192, 255)

def _get_rtc_config() -> RTCConfiguration:
    """Return RTCConfiguration with TURN servers from metered.ca if API key is set."""
    _STUN_ONLY = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    try:
        api_key = st.secrets.get("METERED_API_KEY") or os.environ.get("METERED_API_KEY")
        if not api_key:
            return _STUN_ONLY
        resp = requests.get(
            "https://heart-cvgnal.metered.live/api/v1/turn/credentials",
            params={"apiKey": api_key},
            timeout=5,
        )
        if resp.ok:
            return RTCConfiguration({"iceServers": resp.json()})
    except Exception:
        pass
    return _STUN_ONLY

RTC_CONFIG = _get_rtc_config()

_DEFAULT_STATE: dict = dict(
    score=50, mood="Warming Up", time_left="03:00",
    event_message="", event_id=0, eval_progress=0,
    is_finished=False, final_message="",
    is_calibrating=True, calib_progress=0,
    face_detected=False, pose_detected=False,
    is_leaning=False, is_barrier=False,
    is_looking_away=False, is_tilting=False,
    yaw_deg=0.0, pitch_deg=0.0, event_log=[],
    vlm_available=False, vlm_score=None, vlm_signal="",
    vlm_confidence="", vlm_reasoning="", vlm_age_s=None, vlm_agree=None,
)


def _mood(score: float) -> str:
    label = _MOOD_LABELS[0][1]
    for thr, name in _MOOD_LABELS:
        if score >= thr:
            label = name
    return label


def _final_msg(score: float) -> str:
    if score > 80:  return "It's a Match!  ♥"
    if score >= 50: return "Definitely Something There..."
    return "Let's Just Be Friends..."


# ---------------------------------------------------------------------------
# WebRTC Video Processor
# ---------------------------------------------------------------------------

class HeartCVProcessor(VideoProcessorBase):
    """MediaPipe + AffectionEngine — runs on the Streamlit Cloud server."""

    def __init__(self) -> None:
        self._extractor = FeatureExtractor()
        self._engine    = AffectionEngine()
        self._vlm       = VLMAnalyzer() if _HAS_VLM else None

        self._holistic  = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_hol    = mp.solutions.holistic

        self._lock          = threading.Lock()
        self._state         = dict(_DEFAULT_STATE)

        self._demo_start:   float | None = None
        self._window_start: float | None = None
        self._feature_buf:  list         = []
        self._was_calibrated             = False
        self._last_output                = None
        self._seen_event                 = ""
        self._event_counter              = 0

    # ── API key setter (can be called after init) ─────────────────────────
    def set_api_key(self, key: str) -> bool:
        """Set Anthropic API key and reinitialize VLM. Returns True if VLM is now active."""
        if not _HAS_VLM:
            return False
        os.environ["ANTHROPIC_API_KEY"] = key
        try:
            new_vlm = VLMAnalyzer()
            with self._lock:
                self._vlm = new_vlm
                self._state["vlm_available"] = new_vlm.available
            return new_vlm.available
        except Exception:
            return False

    # ── Thread-safe state accessor ────────────────────────────────────────
    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    # ── Per-frame processing (called by streamlit-webrtc) ─────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        now = time.time()
        ts  = time.monotonic()
        h, w = img.shape[:2]

        # ── MediaPipe inference ───────────────────────────────────────────
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results  = self._holistic.process(rgb)
        features = self._extractor.extract(results, w, h)

        # ── Calibration / scoring ─────────────────────────────────────────
        if not self._was_calibrated:
            output = self._engine.update(features, ts)
            self._last_output = output
            if output.calibrated:
                self._was_calibrated = True
                self._demo_start  = now
                self._window_start = now
        else:
            self._feature_buf.append(features)
            elapsed = now - self._window_start  # type: ignore[operator]
            if elapsed >= _EVAL_WINDOW:
                output = self._engine.batch_evaluate(
                    self._feature_buf, elapsed, ts
                )
                self._last_output = output
                self._feature_buf.clear()
                self._window_start = now
            else:
                output = self._last_output

        # ── Timer ─────────────────────────────────────────────────────────
        if self._demo_start is not None:
            remaining  = max(0.0, _DEMO_DURATION - (now - self._demo_start))
            time_str   = f"{int(remaining) // 60:02d}:{int(remaining) % 60:02d}"
            is_finished = remaining <= 0
            eval_pct    = (
                min(100, int((now - self._window_start) / _EVAL_WINDOW * 100))
                if self._window_start else 0
            )
        else:
            time_str    = "03:00"
            is_finished = False
            eval_pct    = 0

        # ── Overlays ──────────────────────────────────────────────────────
        if results.face_landmarks:
            xs = [lm.x * w for lm in results.face_landmarks.landmark]
            ys = [lm.y * h for lm in results.face_landmarks.landmark]
            cv2.rectangle(
                img,
                (max(0, int(min(xs)) - 8), max(0, int(min(ys)) - 8)),
                (min(w, int(max(xs)) + 8), min(h, int(max(ys)) + 8)),
                _HOT, 2,
            )

        if results.pose_landmarks:
            self._mp_draw.draw_landmarks(
                img, results.pose_landmarks,
                self._mp_hol.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_draw.DrawingSpec(
                    color=_MED, thickness=1, circle_radius=2),
                connection_drawing_spec=self._mp_draw.DrawingSpec(
                    color=_LIGHT, thickness=1),
            )

        # ── VLM ───────────────────────────────────────────────────────────
        if self._vlm and self._was_calibrated:
            self._vlm.maybe_trigger(img, now)

        blended = (
            self._vlm.blend(output.score)
            if self._vlm and output is not None
            else (output.score if output is not None else 50.0)
        )

        # ── Update shared state ───────────────────────────────────────────
        if output is not None:
            evt = output.last_event_text
            if evt and evt != self._seen_event:
                self._seen_event = evt
                self._event_counter += 1

            with self._lock:
                s = self._state
                s["score"]           = round(blended)
                s["mood"]            = _mood(blended)
                s["time_left"]       = time_str
                s["event_message"]   = evt
                s["event_id"]        = self._event_counter
                s["eval_progress"]   = eval_pct
                s["is_finished"]     = is_finished
                s["final_message"]   = _final_msg(blended) if is_finished else ""
                s["is_calibrating"]  = not output.calibrated
                s["calib_progress"]  = round(output.calib_progress * 100)
                s["face_detected"]   = features.face_detected
                s["pose_detected"]   = features.pose_detected
                s["is_leaning"]      = output.is_leaning
                s["is_barrier"]      = output.is_barrier
                s["is_looking_away"] = output.is_looking_away
                s["is_tilting"]      = output.is_tilting
                s["yaw_deg"]         = round(features.yaw_deg, 1)
                s["pitch_deg"]       = round(features.pitch_deg, 1)
                s["event_log"]       = output.event_log[:4]

                if self._vlm:
                    vr = self._vlm.get_result()
                    s["vlm_available"] = self._vlm.available
                    if vr is not None:
                        age = self._vlm.seconds_since_result(ts)
                        s["vlm_score"]      = round(vr.score)
                        s["vlm_signal"]     = vr.dominant_signal
                        s["vlm_confidence"] = vr.confidence
                        s["vlm_reasoning"]  = vr.reasoning
                        s["vlm_age_s"]      = round(age) if age is not None else None
                        s["vlm_agree"]      = abs(output.score - vr.score) < 15

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# CSS (extracted from templates/index.html)
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
  --bg-page:      #FFF8FA;
  --bg-panel:     rgba(255,255,255,0.94);
  --pink-pastel:  #E8799C;
  --pink-light:   #F9C7D8;
  --pink-pale:    #FFF0F5;
  --pink-deep:    #C4587A;
  --pink-muted:   #F4B8D0;
  --text-dark:    #2D2438;
  --text-mid:     #7A6080;
  --text-soft:    #B09AB8;
  --white:        #ffffff;
  --green-sig:    #5cdb6a;
  --border-r:     14px;
}

/* Hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.main .block-container { padding:0.5rem 1rem 1rem !important; max-width:100% !important; background:var(--bg-page); }
section[data-testid="stMain"] { background:var(--bg-page); }

/* ── Floating hearts background ── */
@keyframes floatUp {
  0%   { transform:translateY(0) rotate(0deg);    opacity:0;    }
  8%   { opacity:0.12; }
  88%  { opacity:0.04; }
  100% { transform:translateY(-110vh) rotate(420deg); opacity:0; }
}
.heart-p {
  position:fixed; bottom:-30px; pointer-events:none; user-select:none;
  color:var(--pink-pastel); animation:floatUp linear infinite; z-index:0;
}

/* ── Header ── */
.hcv-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 20px;
  background:rgba(255,255,255,0.92);
  border-bottom:1px solid rgba(244,184,208,0.55);
  border-radius:12px;
  box-shadow:0 2px 16px rgba(232,121,156,0.08);
  margin-bottom:12px;
  backdrop-filter:blur(10px);
  position:relative;
}
.hcv-header::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  border-radius:12px 12px 0 0;
  background:linear-gradient(90deg,transparent,var(--pink-light),var(--pink-pastel),var(--pink-light),transparent);
}
.hcv-title { font-family:'Playfair Display',serif; font-size:1.05rem; letter-spacing:3px; color:var(--text-dark); }
.hcv-timer { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; letter-spacing:4px; color:var(--text-dark); text-align:center; }
.hcv-timer.urgent { color:var(--pink-deep); animation:urgentPulse 0.9s ease-in-out infinite alternate; }
@keyframes urgentPulse {
  from { opacity:1; } to { opacity:0.55; text-shadow:0 0 14px rgba(196,88,122,0.5); }
}
.eval-bar-wrap { width:110px; height:3px; background:rgba(244,184,208,0.30); border-radius:99px; overflow:hidden; margin:4px auto 0; }
.eval-bar-fill { height:100%; background:var(--pink-pastel); border-radius:99px; transition:width 0.5s linear; }
.status-dots { display:flex; gap:14px; font-size:0.68rem; color:var(--text-soft); font-weight:500; }
.dot-row { display:flex; align-items:center; gap:6px; }
.s-dot { width:8px; height:8px; border-radius:50%; background:rgba(176,154,184,0.3); transition:background 0.3s, box-shadow 0.3s; display:inline-block; }
.s-dot.ok  { background:var(--green-sig); box-shadow:0 0 7px var(--green-sig); }
.s-dot.bad { background:var(--pink-pastel); box-shadow:0 0 7px var(--pink-pastel); }

/* ── Cards ── */
.card {
  background:var(--bg-panel);
  border:1.5px solid rgba(244,184,208,0.70);
  border-radius:var(--border-r);
  padding:14px 16px;
  backdrop-filter:blur(14px);
  box-shadow:0 2px 14px rgba(232,121,156,0.08);
  margin-bottom:10px;
}
.card-title { font-size:0.58rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:var(--pink-pastel); margin-bottom:10px; }

/* ── Score ── */
.score-num { font-family:'Playfair Display',serif; font-size:3.2rem; line-height:1; color:var(--pink-deep); display:inline; }
.score-denom { font-size:0.9rem; color:var(--text-mid); font-weight:500; }
.mood-label { font-size:0.75rem; font-style:italic; color:var(--text-mid); margin-bottom:12px; }

/* ── Gauge ── */
.gauge-wrap { position:relative; height:20px; margin-top:4px; }
.gauge-track { position:absolute; inset:0; background:rgba(244,184,208,0.25); border-radius:99px; border:1px solid rgba(232,121,156,0.20); overflow:hidden; }
.gauge-fill { height:100%; background:linear-gradient(90deg,var(--pink-light),var(--pink-pastel),var(--pink-deep)); border-radius:99px; transition:width 0.9s cubic-bezier(.25,.46,.45,.94); box-shadow:0 0 8px rgba(232,121,156,0.30); }
.gauge-heart { position:absolute; top:50%; transform:translateY(-50%); font-size:17px; pointer-events:none; transition:left 0.9s cubic-bezier(.25,.46,.45,.94); filter:drop-shadow(0 0 4px rgba(232,121,156,0.70)); z-index:2; }

/* ── Divider ── */
.div-line { height:1px; background:rgba(244,184,208,0.40); margin:8px 0; }

/* ── Metrics ── */
.m-row { display:flex; justify-content:space-between; align-items:center; font-size:0.72rem; margin-bottom:7px; }
.m-label { color:var(--text-soft); font-weight:500; }
.m-val { font-weight:600; color:var(--text-dark); }
.m-val.warn { color:var(--pink-deep); }
.m-val.ok   { color:var(--green-sig); }

/* ── Signal badges ── */
.signals { display:flex; flex-wrap:wrap; gap:6px; min-height:24px; margin-bottom:2px; }
.badge { font-size:0.60rem; font-weight:700; letter-spacing:1px; padding:3px 10px; border-radius:99px; text-transform:uppercase; }
.badge-green { background:rgba(92,219,106,0.12); color:#3db34a; border:1px solid rgba(92,219,106,0.40); }
.badge-red   { background:rgba(232,121,156,0.12); color:var(--pink-deep); border:1px solid rgba(232,121,156,0.35); }

/* ── Event log ── */
.log-item { font-size:0.66rem; color:var(--text-mid); line-height:1.3; padding:3px 0; border-bottom:1px solid rgba(244,184,208,0.30); }

/* ── VLM ── */
.vlm-offline  { color:var(--text-soft); font-style:italic; font-size:0.68rem; }
.vlm-waiting  { color:var(--pink-pastel); font-style:italic; font-size:0.68rem; }
.vlm-num { font-family:'Playfair Display',serif; font-size:1.5rem; color:var(--pink-deep); line-height:1; }
.vlm-denom { color:var(--text-mid); font-size:0.75rem; }
.vlm-age { color:var(--text-soft); font-size:0.60rem; }
.vlm-mini-track { height:6px; background:rgba(244,184,208,0.25); border-radius:99px; overflow:hidden; margin:6px 0; }
.vlm-mini-fill  { height:100%; background:linear-gradient(90deg,var(--pink-light),var(--pink-pastel)); border-radius:99px; }
.vlm-conf { font-size:0.58rem; font-weight:700; padding:2px 7px; border-radius:99px; text-transform:uppercase; letter-spacing:0.5px; margin-right:6px; }
.vlm-conf.high   { background:rgba(92,219,106,0.15); color:#3db34a; border:1px solid rgba(92,219,106,0.4); }
.vlm-conf.medium { background:rgba(232,121,156,0.10); color:var(--pink-pastel); border:1px solid rgba(232,121,156,0.3); }
.vlm-conf.low    { background:rgba(176,154,184,0.15); color:var(--text-soft); border:1px solid rgba(176,154,184,0.3); }
.vlm-agree      { font-size:0.58rem; font-weight:700; }
.vlm-agree.yes  { color:#3db34a; }
.vlm-agree.no   { color:var(--pink-deep); }
.vlm-reasoning  { color:var(--text-mid); font-style:italic; line-height:1.4; font-size:0.68rem; }

/* ── Calibration card ── */
.calib-card { text-align:center; padding:20px; }
.calib-logo { font-family:'Playfair Display',serif; font-size:1.6rem; letter-spacing:4px; color:var(--text-dark); margin-bottom:8px; }
.calib-sub  { font-size:0.85rem; font-weight:500; letter-spacing:1.5px; color:var(--text-mid); margin-bottom:4px; }
.calib-hint { font-size:0.72rem; color:var(--text-soft); margin-bottom:14px; }
.calib-track { width:100%; max-width:300px; height:10px; background:rgba(244,184,208,0.25); border-radius:99px; border:1px solid rgba(232,121,156,0.25); overflow:hidden; margin:0 auto 8px; }
.calib-fill  { height:100%; background:linear-gradient(90deg,var(--pink-light),var(--pink-pastel)); border-radius:99px; transition:width 0.35s ease; box-shadow:0 0 8px rgba(232,121,156,0.30); }
.calib-pct   { font-size:0.78rem; font-weight:600; color:var(--pink-pastel); }

/* ── Final card ── */
.final-card { text-align:center; padding:40px; background:rgba(255,255,255,0.97); border:1.5px solid rgba(244,184,208,0.70); border-radius:22px; box-shadow:0 8px 40px rgba(232,121,156,0.16); }
.final-title { font-family:'Playfair Display',serif; font-size:2rem; letter-spacing:3px; color:var(--text-dark); margin-bottom:10px; }
.final-score-num { font-family:'Playfair Display',serif; font-size:4.5rem; color:var(--pink-deep); line-height:1; }
.final-msg { font-size:1.05rem; font-style:italic; color:var(--text-mid); margin:8px 0 16px; }
.final-gauge-track { width:80%; max-width:360px; height:16px; background:rgba(244,184,208,0.22); border-radius:99px; overflow:hidden; margin:0 auto; }
.final-gauge-fill  { height:100%; background:linear-gradient(90deg,var(--pink-light),var(--pink-pastel),var(--pink-deep)); border-radius:99px; transition:width 2.2s cubic-bezier(.16,1,.3,1); }
.final-hint { font-size:0.65rem; color:var(--text-soft); margin-top:10px; }

/* ── WebRTC video styling ── */
[data-testid="stVideo"] video,
.stVideo video { border-radius:12px !important; }
</style>
"""

_HEARTS_JS = """
<div id="hbg" style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;"></div>
<script>
(function(){
  const bg=document.getElementById('hbg');
  const g=['♥','♡','✿','❋','✦'];
  for(let i=0;i<20;i++){
    const e=document.createElement('span');
    e.className='heart-p';
    e.textContent=g[i%g.length];
    e.style.left=Math.random()*100+'%';
    e.style.fontSize=(11+Math.random()*15)+'px';
    e.style.animationDuration=(9+Math.random()*13)+'s';
    e.style.animationDelay=(Math.random()*15)+'s';
    bg.appendChild(e);
  }
})();
</script>
"""


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _header_html(state: dict) -> str:
    time_cls  = "urgent" if state["time_left"] < "01:00" else ""
    face_cls  = "ok" if state["face_detected"] else "bad"
    pose_cls  = "ok" if state["pose_detected"] else "bad"
    eval_pct  = state["eval_progress"]
    return f"""
    <div class="hcv-header">
      <div class="hcv-title">HEART ♥ CV-GNAL</div>
      <div>
        <div class="hcv-timer {time_cls}">{state['time_left']}</div>
        <div class="eval-bar-wrap"><div class="eval-bar-fill" style="width:{eval_pct}%"></div></div>
      </div>
      <div class="status-dots">
        <div class="dot-row"><span class="s-dot {face_cls}"></span>Face</div>
        <div class="dot-row"><span class="s-dot {pose_cls}"></span>Pose</div>
      </div>
    </div>
    """


def _calibration_html(state: dict) -> str:
    pct = state["calib_progress"]
    return f"""
    <div class="card calib-card">
      <div class="calib-logo">HEART ♥ CV-GNAL</div>
      <div class="calib-sub">Affection / Interest Analyzer</div>
      <div class="calib-hint">정면을 바라보고 잠시 기다려 주세요.</div>
      <div class="calib-track"><div class="calib-fill" style="width:{pct}%"></div></div>
      <div class="calib-pct">{pct}%</div>
    </div>
    """


def _score_card_html(state: dict) -> str:
    score = state["score"]
    return f"""
    <div class="card">
      <div class="card-title">Heart Gauge</div>
      <div>
        <span class="score-num">{score}</span>
        <span class="score-denom">/ 100</span>
      </div>
      <div class="mood-label">{state['mood']}</div>
      <div class="gauge-wrap">
        <div class="gauge-track">
          <div class="gauge-fill" style="width:{score}%"></div>
        </div>
        <span class="gauge-heart" style="left:{score}%">♥</span>
      </div>
    </div>
    """


def _signals_card_html(state: dict) -> str:
    badges = ""
    if state["is_leaning"]:      badges += '<span class="badge badge-green">▲ Lean In</span>'
    if state["is_tilting"]:      badges += '<span class="badge badge-green">↗ Head Tilt</span>'
    if state["is_looking_away"]: badges += '<span class="badge badge-red">👀 Look Away</span>'
    if state["is_barrier"]:      badges += '<span class="badge badge-red">✖ Barrier</span>'

    yaw   = state["yaw_deg"]
    pitch = state["pitch_deg"]
    yaw_cls   = "warn" if abs(yaw)   > 20 else ""
    pitch_cls = "warn" if abs(pitch) > 15 else ""
    arms_cls  = "warn" if state["is_barrier"] else "ok"
    arms_txt  = "CROSSED" if state["is_barrier"] else "free"

    return f"""
    <div class="card">
      <div class="card-title">Live Signals</div>
      <div class="signals">{badges if badges else '<span style="color:var(--text-soft);font-size:0.68rem;">—</span>'}</div>
      <div class="div-line"></div>
      <div class="m-row"><span class="m-label">Head Yaw</span>
        <span class="m-val {yaw_cls}">{'+' if yaw>=0 else ''}{yaw}°</span></div>
      <div class="m-row"><span class="m-label">Head Pitch</span>
        <span class="m-val {pitch_cls}">{'+' if pitch>=0 else ''}{pitch}°</span></div>
      <div class="m-row"><span class="m-label">Arms</span>
        <span class="m-val {arms_cls}">{arms_txt}</span></div>
    </div>
    """


def _event_log_html(state: dict) -> str:
    items = "".join(
        f'<div class="log-item" style="opacity:{1.0 - i*0.25:.2f}">{e}</div>'
        for i, e in enumerate(state["event_log"])
    )
    return f"""
    <div class="card">
      <div class="card-title">Recent Events</div>
      {items if items else '<span style="color:var(--text-soft);font-size:0.68rem;">No events yet</span>'}
    </div>
    """


def _vlm_card_html(state: dict) -> str:
    if not state["vlm_available"]:
        body = '<div class="vlm-offline">Claude VLM offline<br><span style="font-size:0.60rem">pip install anthropic + set ANTHROPIC_API_KEY</span></div>'
    elif state["vlm_score"] is None:
        body = '<div class="vlm-waiting">Analyzing first frame…</div>'
    else:
        agree_cls  = "yes" if state["vlm_agree"] else "no"
        agree_txt  = "✓ Systems Agree" if state["vlm_agree"] else "⚠ Conflicting"
        conf_cls   = state["vlm_confidence"] or "low"
        age_txt    = f"{state['vlm_age_s']}s ago" if state["vlm_age_s"] is not None else ""
        body = f"""
        <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:4px;">
          <span class="vlm-num">{state['vlm_score']}</span>
          <span class="vlm-denom">/ 100</span>
          <span class="vlm-age" style="margin-left:auto">{age_txt}</span>
        </div>
        <div class="vlm-mini-track"><div class="vlm-mini-fill" style="width:{state['vlm_score']}%"></div></div>
        <div style="margin-bottom:4px;">
          <span class="vlm-conf {conf_cls}">{conf_cls}</span>
          <span class="vlm-agree {agree_cls}">{agree_txt}</span>
        </div>
        <div class="vlm-reasoning">"{state['vlm_reasoning']}"</div>
        """
    return f"""
    <div class="card">
      <div class="card-title">VLM Opinion</div>
      {body}
    </div>
    """


def _final_screen_html(state: dict) -> str:
    score = state["score"]
    return f"""
    <div style="display:flex;justify-content:center;padding:20px 0;">
      <div class="final-card">
        <div class="final-title">Time's Up! ♥</div>
        <div class="final-score-num">{score}</div>
        <div style="font-size:0.9rem;color:var(--text-mid)">/&nbsp;100</div>
        <div class="final-msg">{state['final_message']}</div>
        <div class="final-gauge-track">
          <div class="final-gauge-fill" style="width:{score}%"></div>
        </div>
        <div class="final-hint">Heart CV-gnal · Affection Analysis Complete</div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Heart CV-gnal ♥",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(_CSS, unsafe_allow_html=True)
st.components.v1.html(_HEARTS_JS, height=0)

# ── Session state init ──────────────────────────────────────────────────────
if "last_event_id" not in st.session_state:
    st.session_state["last_event_id"] = 0
if "final_shown" not in st.session_state:
    st.session_state["final_shown"] = False
if "vlm_active" not in st.session_state:
    st.session_state["vlm_active"] = False

# ── Sidebar — API key input ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")
    st.markdown("**Claude VLM Cross-validator**")
    st.caption("Anthropic API 키를 입력하면 Claude가 표정/자세를 추가로 분석합니다.")

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        key="api_key_input",
    )

    apply_clicked = st.button("Apply", key="apply_api_key", use_container_width=True)

    if st.session_state["vlm_active"]:
        st.success("VLM Active")
    else:
        st.info("VLM Offline")

# ── Header placeholder (updated in fragment) ────────────────────────────────
header_ph = st.empty()
header_ph.markdown(_header_html(_DEFAULT_STATE), unsafe_allow_html=True)

# ── Main layout ──────────────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 1], gap="small")

with col_video:
    st.markdown(
        '<div style="border-radius:16px;padding:3px;'
        'background:linear-gradient(135deg,#F9C7D8,#F4B8D0,#E8799C);'
        'box-shadow:0 8px 32px rgba(232,121,156,0.20);display:inline-block;width:100%;">',
        unsafe_allow_html=True,
    )
    webrtc_ctx = webrtc_streamer(
        key="heart-cv",
        video_processor_factory=HeartCVProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Side panel (auto-refreshed every 0.5 s) ──────────────────────────────────
with col_panel:

    @st.fragment(run_every=0.5)
    def side_panel() -> None:
        # Apply API key if button was clicked
        if apply_clicked and api_key_input:
            if webrtc_ctx.video_processor is not None:
                ok = webrtc_ctx.video_processor.set_api_key(api_key_input)
                st.session_state["vlm_active"] = ok
            else:
                # Store for later — processor not ready yet
                st.session_state["pending_api_key"] = api_key_input

        # Apply pending key once processor becomes available
        if (
            "pending_api_key" in st.session_state
            and webrtc_ctx.video_processor is not None
        ):
            pending = st.session_state.pop("pending_api_key")
            ok = webrtc_ctx.video_processor.set_api_key(pending)
            st.session_state["vlm_active"] = ok

        # Get current state
        if webrtc_ctx.video_processor is not None:
            state = webrtc_ctx.video_processor.get_state()
        else:
            state = dict(_DEFAULT_STATE)

        # Update header
        header_ph.markdown(_header_html(state), unsafe_allow_html=True)

        # Toast for new events
        eid = state["event_id"]
        if eid > st.session_state["last_event_id"] and state["event_message"]:
            st.session_state["last_event_id"] = eid
            st.toast(state["event_message"])

        # Final screen
        if state["is_finished"]:
            if not st.session_state["final_shown"]:
                st.session_state["final_shown"] = True
                st.balloons()
            st.markdown(_final_screen_html(state), unsafe_allow_html=True)
            return

        # Calibration
        if state["is_calibrating"] or webrtc_ctx.video_processor is None:
            st.markdown(_calibration_html(state), unsafe_allow_html=True)
            if webrtc_ctx.video_processor is None:
                st.info("카메라를 시작하면 분석이 시작됩니다.")
            return

        # Main panels
        st.markdown(_score_card_html(state),   unsafe_allow_html=True)
        st.markdown(_signals_card_html(state), unsafe_allow_html=True)
        st.markdown(_event_log_html(state),    unsafe_allow_html=True)
        st.markdown(_vlm_card_html(state),     unsafe_allow_html=True)

    side_panel()
