"""
Heart CV-gnal — Flask Web Application
======================================
Wraps the CV evaluation loop in a background thread and serves:

  GET /              → index.html  (the web UI)
  GET /video_feed    → MJPEG stream  (multipart/x-mixed-replace)
  GET /status        → JSON game state

Run from the project root:
    PYTHONPATH=src python app.py
Then open:  http://localhost:5001
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template

# ── Ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from heart_cvgnal.pipelines.vision.affection_engine import AffectionEngine
from heart_cvgnal.pipelines.vision.feature_extractor import FeatureExtractor
from heart_cvgnal.pipelines.vision.vlm_analyzer import VLMAnalyzer

# ── Demo config (mirrors runner.py) ─────────────────────────────────────────
_DEMO_DURATION = 180.0   # 3 minutes
_EVAL_WINDOW   = 5.0     # seconds per scoring window
_CAMERA_INDEX  = 0   # 0 = 맥 내장, 1 = 아이폰 Continuity Camera

# ── Mood labels ──────────────────────────────────────────────────────────────
_MOOD_LABELS = [
    (0,  "Not Feeling It"),
    (20, "Mildly Curious"),
    (40, "Warming Up"),
    (60, "Interested"),
    (75, "Smitten"),
    (88, "Head Over Heels"),
]


def _mood(score: float) -> str:
    label = _MOOD_LABELS[0][1]
    for thr, txt in _MOOD_LABELS:
        if score >= thr:
            label = txt
    return label


def _final_msg(score: float) -> str:
    if score > 80:
        return "It's a Match!  ♥"
    if score >= 50:
        return "Definitely Something There..."
    return "Let's Just Be Friends..."


# ── Shared state (written by CV thread, read by Flask) ───────────────────────
_state_lock = threading.Lock()
_frame_lock = threading.Lock()

_state: dict = {
    "score":           50,
    "mood":            "Warming Up",
    "time_left":       "03:00",
    "event_message":   "",
    "event_id":        0,      # increments on each new distinct event
    "eval_progress":   0,      # 0-100 % of current 5-s window elapsed
    "is_finished":     False,
    "final_message":   "",
    "is_calibrating":  True,
    "calib_progress":  0,      # 0-100
    "face_detected":   False,
    "pose_detected":   False,
    "is_leaning":      False,
    "is_barrier":      False,
    "is_looking_away": False,
    "is_tilting":      False,
    "yaw_deg":         0.0,
    "pitch_deg":       0.0,
    "event_log":       [],
    # VLM cross-validator fields (populated once anthropic is installed + key set)
    "vlm_available":   False,
    "vlm_score":       None,
    "vlm_signal":      "",
    "vlm_confidence":  "",
    "vlm_reasoning":   "",
    "vlm_age_s":       None,   # seconds since last VLM response
    "vlm_agree":       None,   # True / False / None
}

_latest_frame: bytes | None = None

# ── Flask application ─────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _state_lock:
        return jsonify(dict(_state))


def _stream_frames():
    """MJPEG generator — serves the latest encoded frame at up to ~30 fps."""
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)


# ── CV background thread ──────────────────────────────────────────────────────
def _cv_loop(camera_index: int = _CAMERA_INDEX) -> None:
    """
    Runs the full MediaPipe + AffectionEngine pipeline in a background thread.
    All cv2.imshow / cv2.putText / cv2.waitKey calls have been removed.
    Only minimal visual overlays (face bounding box + pose skeleton) are drawn
    directly on the streamed frame.
    """
    global _latest_frame

    extractor = FeatureExtractor()
    engine    = AffectionEngine()
    vlm       = VLMAnalyzer()
    mp_hol    = mp.solutions.holistic
    mp_draw   = mp.solutions.drawing_utils

    # Pink palette (BGR) for the raw-frame overlays
    _HOT   = (147,  20, 255)
    _MED   = (180, 105, 255)
    _LIGHT = (203, 192, 255)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ── Loop-local state ─────────────────────────────────────────────────────
    demo_start:   float | None = None
    window_start: float | None = None
    feature_buf:  list         = []
    last_output                = None
    was_calibrated             = False
    seen_event_text            = ""
    event_counter              = 0
    is_finished                = False

    with mp_hol.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            now   = time.time()
            ts    = time.monotonic()
            h_fr, w_fr = frame.shape[:2]

            # ── MediaPipe inference ───────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # ── Feature extraction ────────────────────────────────────────
            features = extractor.extract(results, w_fr, h_fr)

            # ── Scoring / calibration (identical logic to runner.py) ──────
            if not was_calibrated:
                output = engine.update(features, ts)
                last_output = output
                if output.calibrated:
                    was_calibrated = True
                    demo_start   = now
                    window_start = now
                    print("[INFO] Calibration done — demo timer started.")
            else:
                feature_buf.append(features)
                elapsed = now - window_start  # type: ignore[operator]
                if elapsed >= _EVAL_WINDOW:
                    output = engine.batch_evaluate(feature_buf, elapsed, ts)
                    last_output  = output
                    feature_buf.clear()
                    window_start = now
                else:
                    output = last_output

            # ── Timer ─────────────────────────────────────────────────────
            if demo_start is not None:
                remaining   = max(0.0, _DEMO_DURATION - (now - demo_start))
                mins        = int(remaining) // 60
                secs        = int(remaining) % 60
                time_str    = f"{mins:02d}:{secs:02d}"
                is_finished = remaining <= 0
                eval_pct    = (
                    min(100, int((now - window_start) / _EVAL_WINDOW * 100))  # type: ignore[operator]
                    if window_start is not None else 0
                )
            else:
                time_str  = "03:00"
                eval_pct  = 0

            # ── Minimal visual overlays on the streamed frame ─────────────
            if results.face_landmarks:
                xs = [lm.x * w_fr for lm in results.face_landmarks.landmark]
                ys = [lm.y * h_fr for lm in results.face_landmarks.landmark]
                x1 = max(0, int(min(xs)) - 8)
                y1 = max(0, int(min(ys)) - 8)
                x2 = min(w_fr, int(max(xs)) + 8)
                y2 = min(h_fr, int(max(ys)) + 8)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _HOT, 2)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_hol.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=_MED, thickness=1, circle_radius=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=_LIGHT, thickness=1
                    ),
                )

            # ── VLM: async trigger + blend score ─────────────────────────
            if was_calibrated:
                vlm.maybe_trigger(frame, now)
            blended_score = vlm.blend(output.score) if output is not None else 50.0

            # ── Update shared state ───────────────────────────────────────
            if output is not None:
                score   = blended_score
                new_evt = output.last_event_text

                # Detect new distinct events (text change = new event)
                if new_evt and new_evt != seen_event_text:
                    seen_event_text = new_evt
                    event_counter  += 1

                with _state_lock:
                    _state["score"]           = round(score)
                    _state["mood"]            = _mood(score)
                    _state["time_left"]       = time_str
                    _state["event_message"]   = new_evt
                    _state["event_id"]        = event_counter
                    _state["eval_progress"]   = eval_pct
                    _state["is_finished"]     = is_finished
                    _state["final_message"]   = _final_msg(score) if is_finished else ""
                    _state["is_calibrating"]  = not output.calibrated
                    _state["calib_progress"]  = round(output.calib_progress * 100)
                    _state["face_detected"]   = features.face_detected
                    _state["pose_detected"]   = features.pose_detected
                    _state["is_leaning"]      = output.is_leaning
                    _state["is_barrier"]      = output.is_barrier
                    _state["is_looking_away"] = output.is_looking_away
                    _state["is_tilting"]      = output.is_tilting
                    _state["yaw_deg"]         = round(features.yaw_deg, 1)
                    _state["pitch_deg"]       = round(features.pitch_deg, 1)
                    _state["event_log"]       = output.event_log[:4]

                    # VLM fields
                    vlm_result = vlm.get_result()
                    _state["vlm_available"]  = vlm.available
                    if vlm_result is not None:
                        age = vlm.seconds_since_result(ts)
                        _state["vlm_score"]      = round(vlm_result.score)
                        _state["vlm_signal"]     = vlm_result.dominant_signal
                        _state["vlm_confidence"] = vlm_result.confidence
                        _state["vlm_reasoning"]  = vlm_result.reasoning
                        _state["vlm_age_s"]      = round(age) if age is not None else None
                        _state["vlm_agree"]      = abs(output.score - vlm_result.score) < 15
                    else:
                        _state["vlm_score"]      = None
                        _state["vlm_signal"]     = ""
                        _state["vlm_confidence"] = ""
                        _state["vlm_reasoning"]  = ""
                        _state["vlm_age_s"]      = None
                        _state["vlm_agree"]      = None

            # ── Encode and publish JPEG frame ─────────────────────────────
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            with _frame_lock:
                _latest_frame = buf.tobytes()

            if is_finished:
                break

    cap.release()
    print("[INFO] CV loop finished.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cv_thread = threading.Thread(target=_cv_loop, daemon=True)
    cv_thread.start()

    print("=" * 54)
    print("  Heart CV-gnal  ♥   http://localhost:5001")
    print("  Press  Ctrl+C  to quit")
    print("=" * 54)

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
