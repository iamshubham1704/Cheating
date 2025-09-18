import base64
import io
import json
import os
import time
import threading
from dataclasses import asdict
from typing import Dict, Optional
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory, request, Response
from flask_socketio import SocketIO, emit
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Import behavior analyzer with error handling
try:
    from behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics
    BEHAVIOR_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Behavior analyzer not available: {e}")
    BEHAVIOR_ANALYZER_AVAILABLE = False
    # Create a dummy class for deployment
    class BehaviorAnalyzer:
        def compute_metrics(self, *args, **kwargs):
            return None

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")
# Force threading mode for stability on Windows
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Keep a single analyzer and current candidate info
analyzer = BehaviorAnalyzer() if BEHAVIOR_ANALYZER_AVAILABLE else BehaviorAnalyzer()
_mp_lock = threading.Lock()
_mp_holistic = None  # lazy-initialized global MediaPipe Holistic instance
current_candidate_name: str = ""
_audio_window: list = []  # rolling window of recent RMS/pitch features
_screen_writer = None
_screen_size = None
_screen_last_open_path = None
_last_visual_metrics: Optional[dict] = None
_yolo_model = None
_session_events: list = []  # store tuples of (t, type, payload)


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/hr")
def hr():
    return send_from_directory("templates", "hr.html")


@app.route("/candidate")
def candidate():
    return send_from_directory("templates", "candidate.html")


def generate_analysis_pdf(summary_data):
    """Generate a detailed PDF analysis report for the candidate."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    # Build the PDF content
    story = []
    
    # Title
    story.append(Paragraph("AI Interview Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Candidate Information
    story.append(Paragraph("Candidate Information", heading_style))
    candidate_name = summary_data.get('candidate_name', 'Unknown')
    interview_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    candidate_info = [
        ['Candidate Name:', candidate_name],
        ['Interview Date:', interview_date],
        ['Analysis Duration:', 'Real-time monitoring'],
        ['Report Generated:', datetime.now().strftime("%B %d, %Y at %I:%M %p")]
    ]
    
    candidate_table = Table(candidate_info, colWidths=[2*inch, 4*inch])
    candidate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
    ]))
    
    story.append(candidate_table)
    story.append(Spacer(1, 20))
    
    # Behavioral Analysis Summary
    story.append(Paragraph("Behavioral Analysis Summary", heading_style))
    
    avg_confidence = summary_data.get('avg_confidence', 0.0)
    avg_cheating = summary_data.get('avg_cheating', 0.0)
    avg_nervous = summary_data.get('avg_nervous', 0.0)
    alerts_count = summary_data.get('alerts_count', 0)
    
    # Convert to percentages
    confidence_pct = avg_confidence * 100
    cheating_pct = avg_cheating * 100
    nervous_pct = avg_nervous * 100
    
    # Determine confidence level
    if confidence_pct >= 80:
        confidence_level = "High"
    elif confidence_pct >= 60:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Determine cheating risk
    if cheating_pct >= 70:
        cheating_risk = "High Risk"
    elif cheating_pct >= 40:
        cheating_risk = "Medium Risk"
    else:
        cheating_risk = "Low Risk"
    
    # Determine nervousness level
    if nervous_pct >= 70:
        nervous_level = "High"
    elif nervous_pct >= 40:
        nervous_level = "Medium"
    else:
        nervous_level = "Low"
    
    metrics_data = [
        ['Metric', 'Score', 'Percentage', 'Assessment'],
        ['Confidence Level', f"{confidence_pct:.1f}%", f"{confidence_pct:.1f}%", confidence_level],
        ['Cheating Risk', f"{cheating_pct:.1f}%", f"{cheating_pct:.1f}%", cheating_risk],
        ['Nervousness', f"{nervous_pct:.1f}%", f"{nervous_pct:.1f}%", nervous_level],
        ['Security Alerts', str(alerts_count), 'N/A', 'High' if alerts_count > 5 else 'Medium' if alerts_count > 2 else 'Low']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('BACKGROUND', (1, 1), (1, -1), colors.beige),
        ('BACKGROUND', (2, 1), (2, -1), colors.beige),
        ('BACKGROUND', (3, 1), (3, -1), colors.beige),
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Complete Transcript Section
    story.append(Paragraph("Complete Interview Transcript", heading_style))
    
    # Get questions and answers from session events
    qa_events = []
    session_events = summary_data.get('session_events', [])
    for t, kind, payload in session_events:
        if kind == "transcript":
            qa_events.append((t, payload))
    
    if qa_events:
        story.append(Paragraph(f"Total communication entries: {len(qa_events)}", normal_style))
        story.append(Spacer(1, 12))
        
        # Show sample transcripts (first 10)
        for i, (timestamp, transcript) in enumerate(qa_events[:10], 1):
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            text = transcript.get('text', '')[:200] + ('...' if len(transcript.get('text', '')) > 200 else '')
            story.append(Paragraph(f"<b>Entry {i} ({time_str}):</b> {text}", normal_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No communication data available for this interview session.", normal_style))
        story.append(Spacer(1, 12))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    
    recommendations = []
    if confidence_pct < 60:
        recommendations.append("• Consider additional interview preparation or coaching to boost confidence")
    if cheating_pct > 40:
        recommendations.append("• Review interview policies and consider additional monitoring for future sessions")
    if nervous_pct > 60:
        recommendations.append("• Provide stress management resources or consider a more relaxed interview format")
    if alerts_count > 5:
        recommendations.append("• Implement stricter monitoring protocols for future interviews")
    if not recommendations:
        recommendations.append("• Candidate demonstrated good overall performance with no major concerns")
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
        story.append(Spacer(1, 6))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"""
    <i>This report was generated automatically by AI Interview Analysis System on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}. 
    The analysis is based on real-time behavioral monitoring and should be used as supplementary information 
    in the hiring decision process.</i>
    """
    story.append(Paragraph(footer_text, normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    """Generate and return PDF analysis report."""
    try:
        summary_data = request.get_json()
        if not summary_data:
            return Response("No data provided", status=400)
        
        pdf_content = generate_analysis_pdf(summary_data)
        
        return Response(
            pdf_content,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename=interview_analysis_{summary_data.get("candidate_name", "candidate")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            }
        )
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return Response(f"Error generating PDF: {str(e)}", status=500)


@socketio.on("transcript")
def on_transcript(data):
    text = (data or {}).get("text") or ""
    if not text:
        return
    rec = {"candidate_name": current_candidate_name, "text": text, "timestamp": time.time()}
    _session_events.append((time.time(), "transcript", rec))
    emit("transcript", rec, broadcast=True)


@socketio.on("end_interview")
def on_end_interview(_: dict):
    # Compute summary
    times, confs, cheats, nervs = [], [], [], []
    transcripts = []
    alerts = []
    for t, kind, payload in _session_events:
        if kind == "meters":
            times.append(t)
            confs.append(float(payload.get("confidence_meter", 0.0)))
            cheats.append(float(payload.get("cheating_meter", 0.0)))
            nervs.append(float(payload.get("nervous_meter", 0.0)))
        elif kind == "transcript":
            transcripts.append(payload)
        elif kind == "alert":
            alerts.append(payload)

    avg_conf = float(np.mean(confs)) if confs else 0.0
    avg_cheat = float(np.mean(cheats)) if cheats else 0.0
    avg_nerv = float(np.mean(nervs)) if nervs else 0.0

    summary = {
        "candidate_name": current_candidate_name or "-",
        "avg_confidence": avg_conf,
        "avg_cheating": avg_cheat,
        "avg_nervous": avg_nerv,
        "alerts_count": len(alerts),
        "transcripts": transcripts,
        "alerts": alerts,
        "session_events": _session_events.copy(),
    }
    emit("interview_summary", summary, broadcast=True)
    # Clear for next session
    _session_events.clear()


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
    kind = (data or {}).get("kind", "webcam")
    data_url = (data or {}).get("image")

    frame_bgr = _decode_frame(data_url)
    if frame_bgr is None:
        return

    # Simplified frame processing for deployment
    metrics_dict = None
    if kind == "webcam" and BEHAVIOR_ANALYZER_AVAILABLE:
        try:
            # Basic frame processing without heavy dependencies
            pass
        except Exception as e:
            print(f"Frame processing error: {e}")

    # Re-encode a lightweight JPEG to forward to HR
    try:
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
        out_data_url = f"data:image/jpeg;base64,{b64}"

        payload = {"kind": kind, "image": out_data_url, "candidate_name": current_candidate_name}
        if metrics_dict is not None:
            payload["metrics"] = metrics_dict

        try:
            socketio.emit("frame", payload, include_self=False)
        except Exception:
            pass
    except Exception as e:
        print(f"Frame encoding error: {e}")


@socketio.on("audio")
def on_audio(data):
    try:
        b64 = (data or {}).get("b64")
        sample_rate = int((data or {}).get("sampleRate", 16000))
        if not b64:
            return
        raw = base64.b64decode(b64)
        pcm = np.frombuffer(raw, dtype=np.int16)
        
        # Basic audio processing
        if len(pcm) > 0:
            rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
            features = {"rms": rms, "candidate_name": current_candidate_name}
            
            try:
                socketio.emit("audio_metrics", features, include_self=False)
            except Exception:
                pass
    except Exception:
        return


if __name__ == "__main__":
    # Run with Werkzeug in threading mode (Windows-friendly)
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
