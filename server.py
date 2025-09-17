import base64
import io
import json
import os
import time
import threading
from dataclasses import asdict
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit

from behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")
# Use default async mode (threading) for broad compatibility on Windows
socketio = SocketIO(app, cors_allowed_origins="*")

# Keep a single analyzer and current candidate info
analyzer = BehaviorAnalyzer()
_mp_lock = threading.Lock()
_mp_holistic = None  # lazy-initialized global MediaPipe Holistic instance
current_candidate_name: str = ""
_audio_window: list = []  # rolling window of recent RMS/pitch features
_screen_writer = None
_screen_size = None
_screen_last_open_path = None
_last_visual_metrics: Optional[dict] = None


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/hr")
def hr():
    return send_from_directory("templates", "hr.html")


@app.route("/candidate")
def candidate():
    return send_from_directory("templates", "candidate.html")


@socketio.on("candidate_info")
def on_candidate_info(data):
    global current_candidate_name
    name = (data or {}).get("name") or ""
    current_candidate_name = str(name)
    emit("candidate_info", {"name": current_candidate_name}, broadcast=True)


def _decode_frame(data_url: str) -> Optional[np.ndarray]:
    if not data_url or not data_url.startswith("data:image/"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64)
        image = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


@socketio.on("frame")
def on_frame(data):
    kind = (data or {}).get("kind", "webcam")  # webcam or screen
    data_url = (data or {}).get("image")

    frame_bgr = _decode_frame(data_url)
    if frame_bgr is None:
        return

    # Run MediaPipe only for webcam frames; for screen frames we just relay
    metrics_dict = None
    if kind == "webcam":
        # Lazy import of mediapipe to avoid GPU init at module load
        import mediapipe as mp  # type: ignore

        global _mp_holistic
        if _mp_holistic is None:
            _mp_holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

        # Serialize process() calls to ensure monotonically increasing timestamps
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with _mp_lock:
            results = _mp_holistic.process(frame_rgb)
        metrics = analyzer.compute_metrics(
            frame_bgr,
            results.pose_landmarks,
            results.face_landmarks,
            results.left_hand_landmarks,
            results.right_hand_landmarks,
        )
        metrics_dict = asdict(metrics)

    # Optionally save screen stream to MP4
    if kind == "screen":
        global _screen_writer, _screen_size, _screen_last_open_path
        h, w = frame_bgr.shape[:2]
        if _screen_writer is None or _screen_size != (w, h):
            # Open a new writer in ./captures with timestamp
            os.makedirs("captures", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captures/screen_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            _screen_writer = cv2.VideoWriter(filename, fourcc, 12.0, (w, h))
            _screen_size = (w, h)
            _screen_last_open_path = filename
        if _screen_writer is not None:
            _screen_writer.write(frame_bgr)

    # Re-encode a lightweight JPEG to forward to HR
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
    out_data_url = f"data:image/jpeg;base64,{b64}"

    payload = {"kind": kind, "image": out_data_url, "candidate_name": current_candidate_name}
    if metrics_dict is not None:
        payload["metrics"] = metrics_dict
        # Attach meters combining audio + visual
        audio_avg = None
        if _audio_window:
            keys = _audio_window[-1].keys()
            audio_avg = {k: float(np.mean([f[k] for f in _audio_window if k in f])) for k in keys}
        meters = _compute_confidence_and_cheating(metrics_dict, audio_avg)
        payload["meters"] = meters
        global _last_visual_metrics
        _last_visual_metrics = metrics_dict

    socketio.emit("frame", payload, include_self=False)


def _analyze_audio_chunk(pcm16: np.ndarray, sample_rate: int) -> dict:
    """Compute simple audio features: RMS loudness and pitch estimate.
    Returns a dict with 'rms', 'pitch_hz' and derived 'nervous_score' and 'cheating_score' heuristics.
    """
    if pcm16.size == 0:
        return {}
    x = pcm16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))

    # Pitch via autocorrelation (very rough)
    max_freq = 400
    min_freq = 75
    max_lag = int(sample_rate / min_freq)
    min_lag = int(sample_rate / max_freq)
    if max_lag >= x.size:
        return {"rms": rms, "pitch_hz": 0.0}
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr[:min_lag] = 0
    if corr.size <= min_lag + 1:
        pitch = 0.0
    else:
        lag = np.argmax(corr[min_lag:max_lag]) + min_lag
        pitch = float(sample_rate / lag) if lag > 0 else 0.0

    # Heuristic scores (placeholder logic):
    nervous_score = float(np.clip((rms - 0.02) * 25.0, 0.0, 1.0))
    cheating_score = float(np.clip((pitch - 280.0) / 120.0, 0.0, 1.0))

    return {"rms": rms, "pitch_hz": pitch, "nervous_score": nervous_score, "cheating_score": cheating_score}


def _compute_confidence_and_cheating(metrics: Optional[dict], audio_avg: Optional[dict]) -> dict:
    """Combine visual and audio signals into simple meters in [0,1]."""
    visual_conf = 0.5
    cheating = 0.0
    nervous = 0.0

    if metrics:
        # Visual confidence higher when posture upright, low fidget, stable eyes open
        posture_bonus = 0.2 if metrics.get("posture") == "upright" else (0.1 if metrics.get("posture") == "leaning" else 0.0)
        fidget = float(metrics.get("fidget_score", 0.0))
        eyes_closed = float(metrics.get("eyes_closed_ratio", 0.0))
        hand_face = 0.15 if metrics.get("hand_to_face") else 0.0
        visual_conf = float(np.clip(0.7 + posture_bonus - 8.0 * fidget - eyes_closed * 0.3 - hand_face, 0.0, 1.0))
        cheating = max(cheating, 0.5 if metrics.get("hand_to_face") else 0.0)

    if audio_avg:
        nervous = float(np.clip(audio_avg.get("nervous_score", 0.0), 0.0, 1.0))
        cheating = max(cheating, float(np.clip(audio_avg.get("cheating_score", 0.0), 0.0, 1.0)))

    return {"confidence_meter": visual_conf, "cheating_meter": cheating, "nervous_meter": nervous}


@socketio.on("audio")
def on_audio(data):
    try:
        b64 = (data or {}).get("b64")
        sample_rate = int((data or {}).get("sampleRate", 16000))
        if not b64:
            return
        raw = base64.b64decode(b64)
        pcm = np.frombuffer(raw, dtype=np.int16)
        features = _analyze_audio_chunk(pcm, sample_rate)
        if not features:
            return
        # Maintain short rolling average for smoothing
        _audio_window.append(features)
        if len(_audio_window) > 20:
            _audio_window.pop(0)
        avg = {k: float(np.mean([f[k] for f in _audio_window if k in f])) for k in features.keys()}
        avg["candidate_name"] = current_candidate_name
        socketio.emit("audio_metrics", avg, include_self=False)
        # Also emit updated meters if we have last visual metrics
        if _last_visual_metrics is not None:
            meters = _compute_confidence_and_cheating(_last_visual_metrics, avg)
            socketio.emit("meters", meters, include_self=False)
    except Exception:
        return


if __name__ == "__main__":
    # Use eventlet for WebSocket
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
