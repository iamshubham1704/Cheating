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

from behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")
# Force threading mode for stability on Windows
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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
        confidence_color = colors.green
    elif confidence_pct >= 60:
        confidence_level = "Medium"
        confidence_color = colors.orange
    else:
        confidence_level = "Low"
        confidence_color = colors.red
    
    # Determine cheating risk
    if cheating_pct >= 70:
        cheating_risk = "High Risk"
        cheating_color = colors.red
    elif cheating_pct >= 40:
        cheating_risk = "Medium Risk"
        cheating_color = colors.orange
    else:
        cheating_risk = "Low Risk"
        cheating_color = colors.green
    
    # Determine nervousness level
    if nervous_pct >= 70:
        nervous_level = "High"
        nervous_color = colors.red
    elif nervous_pct >= 40:
        nervous_level = "Medium"
        nervous_color = colors.orange
    else:
        nervous_level = "Low"
        nervous_color = colors.green
    
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
    
    # Detailed Analysis
    story.append(Paragraph("Detailed Analysis", heading_style))
    
    # Confidence Analysis
    confidence_analysis = f"""
    <b>Confidence Assessment:</b><br/>
    The candidate demonstrated a {confidence_level.lower()} confidence level ({confidence_pct:.1f}%) throughout the interview. 
    This indicates {'strong' if confidence_pct >= 80 else 'moderate' if confidence_pct >= 60 else 'limited'} 
    self-assurance and {'excellent' if confidence_pct >= 80 else 'good' if confidence_pct >= 60 else 'room for improvement in'} 
    professional presence.
    """
    story.append(Paragraph(confidence_analysis, normal_style))
    story.append(Spacer(1, 12))
    
    # Cheating Risk Analysis
    cheating_analysis = f"""
    <b>Integrity Assessment:</b><br/>
    The cheating risk was assessed as {cheating_risk.lower()} ({cheating_pct:.1f}%). 
    {'This indicates a high level of integrity and professional conduct.' if cheating_pct < 40 else 
    'Some concerning behaviors were detected that may warrant further investigation.' if cheating_pct >= 70 else 
    'The candidate generally maintained professional standards with minor concerns noted.'}
    """
    story.append(Paragraph(cheating_analysis, normal_style))
    story.append(Spacer(1, 12))
    
    # Nervousness Analysis
    nervous_analysis = f"""
    <b>Stress Management:</b><br/>
    The candidate showed {nervous_level.lower()} levels of nervousness ({nervous_pct:.1f}%). 
    {'This suggests excellent stress management and composure.' if nervous_pct < 40 else 
    'The candidate may benefit from additional support or preparation for high-pressure situations.' if nervous_pct >= 70 else 
    'The candidate demonstrated reasonable composure with some stress indicators.'}
    """
    story.append(Paragraph(nervous_analysis, normal_style))
    story.append(Spacer(1, 12))
    
    # Security Alerts
    if alerts_count > 0:
        alert_analysis = f"""
        <b>Security Monitoring:</b><br/>
        {alerts_count} security alert{'s were' if alerts_count != 1 else ' was'} triggered during the interview, 
        {'indicating potential policy violations that require attention.' if alerts_count > 5 else 
        'suggesting some concerning behaviors that should be reviewed.' if alerts_count > 2 else 
        'showing minimal security concerns.'}
        """
        story.append(Paragraph(alert_analysis, normal_style))
        story.append(Spacer(1, 12))
    
    # Questions and Answers Section
    story.append(Paragraph("Interview Questions and Answers", heading_style))
    
    # Get questions and answers from session events
    qa_events = []
    session_events = summary_data.get('session_events', [])
    for t, kind, payload in session_events:
        if kind == "transcript":
            qa_events.append((t, payload))
    
    if qa_events:
        story.append(Paragraph(f"Total communication entries: {len(qa_events)}", normal_style))
        story.append(Spacer(1, 12))
        
        # Group transcripts by time intervals to identify Q&A patterns
        qa_pairs = []
        current_qa = {"question": "", "answer": "", "timestamp": None}
        
        # Question indicators
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you', 'would you', 'tell me', 'describe', 'explain']
        
        for i, (timestamp, transcript) in enumerate(qa_events):
            text = transcript.get('text', '').strip()
            if not text:
                continue
                
            # Check if this looks like a question
            is_question = (text.endswith('?') or 
                          any(indicator in text.lower() for indicator in question_indicators) or
                          text.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Can you', 'Could you', 'Would you', 'Tell me', 'Describe', 'Explain')))
            
            if is_question:
                # Save previous Q&A if exists
                if current_qa["question"] and current_qa["answer"]:
                    qa_pairs.append(current_qa.copy())
                
                # Start new Q&A
                current_qa = {
                    "question": text,
                    "answer": "",
                    "timestamp": timestamp
                }
            else:
                # This is likely an answer or continuation
                if current_qa["question"]:
                    current_qa["answer"] += (" " + text) if current_qa["answer"] else text
                else:
                    # If no question context, treat as standalone response
                    if not qa_pairs or qa_pairs[-1]["answer"]:
                        qa_pairs.append({
                            "question": "Candidate Statement",
                            "answer": text,
                            "timestamp": timestamp
                        })
                    else:
                        # Append to last answer
                        qa_pairs[-1]["answer"] += (" " + text) if qa_pairs[-1]["answer"] else text
        
        # Add the last Q&A if exists
        if current_qa["question"]:
            if current_qa["answer"]:
                qa_pairs.append(current_qa)
            else:
                # Question without answer - add as standalone
                qa_pairs.append({
                    "question": current_qa["question"],
                    "answer": "No response recorded",
                    "timestamp": current_qa["timestamp"]
                })
        
        # Display Q&A pairs
        if qa_pairs:
            story.append(Paragraph(f"Identified {len(qa_pairs)} question-answer interactions:", normal_style))
            story.append(Spacer(1, 12))
            
            for i, qa in enumerate(qa_pairs[:15], 1):  # Limit to first 15 Q&A pairs
                timestamp = datetime.fromtimestamp(qa["timestamp"]).strftime("%H:%M:%S")
                
                # Question
                question_text = qa['question']
                if len(question_text) > 200:
                    question_text = question_text[:200] + "..."
                
                story.append(Paragraph(f"<b>Q{i} ({timestamp}):</b> {question_text}", normal_style))
                story.append(Spacer(1, 6))
                
                # Answer (truncate if too long)
                answer_text = qa['answer']
                if len(answer_text) > 400:
                    answer_text = answer_text[:400] + "..."
                
                story.append(Paragraph(f"<b>Answer:</b> {answer_text}", normal_style))
                story.append(Spacer(1, 12))
            
            # Show remaining count if there are more
            if len(qa_pairs) > 15:
                story.append(Paragraph(f"... and {len(qa_pairs) - 15} more interactions", normal_style))
                story.append(Spacer(1, 12))
        else:
            # Fallback: show raw transcripts
            story.append(Paragraph("Communication Log (No clear Q&A pattern detected):", normal_style))
            story.append(Spacer(1, 6))
            
            for i, (timestamp, transcript) in enumerate(qa_events[:15], 1):
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                text = transcript.get('text', '')[:250] + ('...' if len(transcript.get('text', '')) > 250 else '')
                story.append(Paragraph(f"<b>Entry {i} ({time_str}):</b> {text}", normal_style))
                story.append(Spacer(1, 6))
            
            # Show remaining count if there are more
            if len(qa_events) > 15:
                story.append(Paragraph(f"... and {len(qa_events) - 15} more entries", normal_style))
                story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No communication data available for this interview session.", normal_style))
        story.append(Spacer(1, 12))
    
    # Complete Transcript Section
    story.append(Paragraph("Complete Interview Transcript", heading_style))
    
    if qa_events:
        # Calculate transcript statistics
        total_entries = len(qa_events)
        interviewer_entries = 0
        candidate_entries = 0
        total_words = 0
        total_chars = 0
        
        for timestamp, transcript in qa_events:
            text = transcript.get('text', '').strip()
            if text:
                total_words += len(text.split())
                total_chars += len(text)
                
                # Count speaker types
                if (text.endswith('?') or 
                    any(indicator in text.lower() for indicator in ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you', 'would you', 'tell me', 'describe', 'explain']) or
                    text.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Can you', 'Could you', 'Would you', 'Tell me', 'Describe', 'Explain'))):
                    interviewer_entries += 1
                else:
                    candidate_entries += 1
        
        # Add transcript statistics
        story.append(Paragraph("Transcript Statistics:", normal_style))
        story.append(Spacer(1, 8))
        
        stats_data = [
            ['Metric', 'Count', 'Percentage'],
            ['Total Entries', str(total_entries), '100%'],
            ['Interviewer Entries', str(interviewer_entries), f"{(interviewer_entries/total_entries*100):.1f}%" if total_entries > 0 else "0%"],
            ['Candidate Entries', str(candidate_entries), f"{(candidate_entries/total_entries*100):.1f}%" if total_entries > 0 else "0%"],
            ['Total Words', str(total_words), 'N/A'],
            ['Total Characters', str(total_chars), 'N/A'],
            ['Avg Words per Entry', f"{total_words/total_entries:.1f}" if total_entries > 0 else "0", 'N/A']
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('BACKGROUND', (1, 1), (1, -1), colors.beige),
            ('BACKGROUND', (2, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph(f"Full communication log with {len(qa_events)} entries:", normal_style))
        story.append(Spacer(1, 12))
        
        # Create a table for better transcript formatting
        transcript_data = [['Time', 'Speaker', 'Content']]
        
        for i, (timestamp, transcript) in enumerate(qa_events, 1):
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            text = transcript.get('text', '').strip()
            
            # Determine speaker (this is a simplified approach - in a real system you'd have speaker identification)
            speaker = "Interviewer" if (text.endswith('?') or 
                                     any(indicator in text.lower() for indicator in ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you', 'would you', 'tell me', 'describe', 'explain']) or
                                     text.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Can you', 'Could you', 'Would you', 'Tell me', 'Describe', 'Explain'))) else "Candidate"
            
            # Truncate very long content for table display
            display_text = text[:150] + ('...' if len(text) > 150 else '')
            transcript_data.append([time_str, speaker, display_text])
        
        # Create transcript table
        transcript_table = Table(transcript_data, colWidths=[1*inch, 1.2*inch, 4.3*inch])
        transcript_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('BACKGROUND', (1, 1), (1, -1), colors.beige),
            ('BACKGROUND', (2, 1), (2, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(transcript_table)
        story.append(Spacer(1, 20))
        
        # Add detailed transcript entries for better readability
        story.append(Paragraph("Detailed Transcript Entries:", normal_style))
        story.append(Spacer(1, 8))
        
        # Limit detailed entries to prevent extremely long PDFs
        max_detailed_entries = 50
        entries_to_show = qa_events[:max_detailed_entries] if len(qa_events) > max_detailed_entries else qa_events
        
        for i, (timestamp, transcript) in enumerate(entries_to_show, 1):
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            text = transcript.get('text', '').strip()
            
            if text:  # Only include non-empty entries
                # Determine speaker
                speaker = "Interviewer" if (text.endswith('?') or 
                                         any(indicator in text.lower() for indicator in ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you', 'would you', 'tell me', 'describe', 'explain']) or
                                         text.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Can you', 'Could you', 'Would you', 'Tell me', 'Describe', 'Explain'))) else "Candidate"
                
                # Create speaker-specific styling
                if speaker == "Interviewer":
                    speaker_style = ParagraphStyle(
                        'InterviewerStyle',
                        parent=normal_style,
                        leftIndent=20,
                        rightIndent=20,
                        spaceAfter=6,
                        borderColor=colors.blue,
                        borderWidth=1,
                        borderPadding=8,
                        backColor=colors.lightblue
                    )
                    speaker_text = f"<b>[{speaker}]</b> {text}"
                else:
                    speaker_style = ParagraphStyle(
                        'CandidateStyle',
                        parent=normal_style,
                        leftIndent=20,
                        rightIndent=20,
                        spaceAfter=6,
                        borderColor=colors.green,
                        borderWidth=1,
                        borderPadding=8,
                        backColor=colors.lightgreen
                    )
                    speaker_text = f"<b>[{speaker}]</b> {text}"
                
                # Add timestamp and content
                story.append(Paragraph(f"<b>{time_str}</b>", normal_style))
                story.append(Paragraph(speaker_text, speaker_style))
                story.append(Spacer(1, 4))
        
        # Add note if there are more entries
        if len(qa_events) > max_detailed_entries:
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<i>Note: Showing first {max_detailed_entries} entries out of {len(qa_events)} total. Complete transcript is available in the summary table above.</i>", normal_style))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No transcript data available for this interview session.", normal_style))
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
        "session_events": _session_events.copy(),  # Include all session events for Q&A analysis
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

        # Run YOLO object detection (lightweight) and attach suspicious objects
        global _yolo_model
        if _yolo_model is None:
            try:
                from ultralytics import YOLO  # type: ignore
                _yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                _yolo_model = False  # mark as unavailable
        detected_objects = []
        suspicious_objects = []
        if _yolo_model not in (None, False):
            try:
                results_yolo = _yolo_model.predict(source=frame_bgr, imgsz=480, conf=0.35, verbose=False)
                for res in results_yolo:
                    boxes = getattr(res, 'boxes', None)
                    names = getattr(res, 'names', None) or getattr(_yolo_model, 'names', {})
                    if boxes is None:
                        continue
                    for b in boxes:
                        cls_id = int(b.cls[0].cpu().numpy())
                        conf = float(b.conf[0].cpu().numpy())
                        name = names.get(cls_id, str(cls_id))
                        detected_objects.append({"class": name, "confidence": conf})
                        if name in ("cell phone", "book", "laptop", "tv", "remote") and conf >= 0.4:
                            suspicious_objects.append({"class": name, "confidence": conf})
            except Exception:
                pass
        if detected_objects:
            metrics_dict["objects"] = detected_objects
        if suspicious_objects:
            metrics_dict["suspicious_objects"] = suspicious_objects

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

        # If suspicious objects found (YOLO or hand-object heuristic), increase cheating and emit alerts
        suspicious_entries = []
        if metrics_dict.get("suspicious_objects"):
            suspicious_entries.extend(metrics_dict.get("suspicious_objects", []))
        if bool(metrics_dict.get("holding_object")) and metrics_dict.get("object_kind") in ("phone", "paper"):
            suspicious_entries.append({"class": metrics_dict.get("object_kind"), "confidence": float(metrics_dict.get("object_confidence", 0.5))})

        if suspicious_entries:
            # Do not mutate cheating here; unified computation already accounts for objects
            # Emit alerts per top entry
            top = max(suspicious_entries, key=lambda x: x.get("confidence", 0.0))
            print(f"[ALERT] Suspicious object detected: {top.get('class')} conf={top.get('confidence',0):.2f} name={current_candidate_name}")
            alert_msg = {
                "type": "suspicious_object",
                "object": top.get("class"),
                "confidence": float(top.get("confidence", 0.0)),
                "candidate_name": current_candidate_name,
                "timestamp": time.time(),
            }
            try:
                socketio.emit("alert_hr", alert_msg, include_self=False)
                socketio.emit("alert_candidate", {"message": f"Please remove {alert_msg['object']}. Cheating is prohibited.", "severity": "warning"})
            except Exception:
                pass
            _session_events.append((time.time(), "alert", alert_msg))

    try:
        socketio.emit("frame", payload, include_self=False)
    except Exception:
        pass
    # Persist meters for summary
    if payload.get("meters"):
        _session_events.append((time.time(), "meters", payload["meters"]))


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
    """Combine visual posture/fidget, audio cues, and object detections into meters in [0,1]."""
    visual_conf = 0.5
    nervous = 0.0

    cheating_components = []

    if metrics:
        posture = metrics.get("posture")
        fidget = float(metrics.get("fidget_score", 0.0))
        eyes_closed = float(metrics.get("eyes_closed_ratio", 0.0))
        hand_to_face = bool(metrics.get("hand_to_face", False))

        # Confidence: upright posture, low fidget, eyes open
        posture_bonus = 0.22 if posture == "upright" else (0.08 if posture == "leaning" else -0.08 if posture == "slouching" else 0.0)
        visual_conf = float(np.clip(0.65 + posture_bonus - 7.0 * fidget - eyes_closed * 0.35 - (0.1 if hand_to_face else 0.0), 0.0, 1.0))

        # Cheating components
        if hand_to_face:
            cheating_components.append(0.25)
        # Posture contribution if slouching/leaning with high fidget
        if posture in ("slouching", "leaning"):
            cheating_components.append(min(0.2, 4.0 * fidget))

        # Object detections
        obj_conf = 0.0
        if metrics.get("suspicious_objects"):
            obj_conf = float(max([o.get("confidence", 0.0) for o in metrics.get("suspicious_objects", [])] or [0.0]))
        if bool(metrics.get("holding_object")) and metrics.get("object_kind") in ("phone", "paper"):
            obj_conf = max(obj_conf, float(metrics.get("object_confidence", 0.5)))
        if obj_conf > 0.0:
            cheating_components.append(min(0.95, 0.6 + 0.35 * obj_conf))

    if audio_avg:
        nervous = float(np.clip(audio_avg.get("nervous_score", 0.0), 0.0, 1.0))
        audio_cheat = float(np.clip(audio_avg.get("cheating_score", 0.0), 0.0, 1.0))
        if audio_cheat > 0.0:
            cheating_components.append(0.4 * audio_cheat)

    # Combine cheating components using 1 - product(1 - x)
    prod = 1.0
    for c in cheating_components:
        prod *= (1.0 - float(np.clip(c, 0.0, 1.0)))
    cheating = float(np.clip(1.0 - prod, 0.0, 1.0))

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
        try:
            socketio.emit("audio_metrics", avg, include_self=False)
        except Exception:
            pass
        # Also emit updated meters if we have last visual metrics
        if _last_visual_metrics is not None:
            meters = _compute_confidence_and_cheating(_last_visual_metrics, avg)
            try:
                socketio.emit("meters", meters, include_self=False)
            except Exception:
                pass
            _session_events.append((time.time(), "meters", meters))
    except Exception:
        return


if __name__ == "__main__":
    # Run with Werkzeug in threading mode (Windows-friendly)
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
