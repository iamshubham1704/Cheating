# AI Interview Analysis System - Flow & Tech Stack

## 🏗️ **SYSTEM ARCHITECTURE OVERVIEW**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI INTERVIEW ANALYSIS SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CANDIDATE     │    │   HR DASHBOARD  │    │   SERVER        │
│   INTERFACE     │    │   INTERFACE     │    │   BACKEND       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Webcam Stream │    │ • Live Analysis │    │ • Flask Server  │
│ • Screen Share  │    │ • AI Metrics    │    │ • Socket.IO     │
│ • Audio Capture │    │ • Follow-up Q's │    │ • MediaPipe     │
│ • Name Input    │    │ • Real-time UI  │    │ • OpenCV        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 **COMPLETE SYSTEM FLOW**

### **1. INITIALIZATION PHASE**
```
User Access → Server Start → Route Selection → Interface Loading
```

**Tech Stack:**
- **Backend:** Flask (Python web framework)
- **Routes:** `/`, `/candidate`, `/hr` 
- **Templates:** HTML5 + CSS3 + JavaScript
- **WebSocket:** Socket.IO for real-time communication

### **2. CANDIDATE INTERFACE FLOW**
```
Candidate Page Load → Name Input → Start Session → Media Capture → Stream to Server
```

**Detailed Steps:**
1. **Page Load** (`/candidate`)
   - Load candidate.html template
   - Initialize Socket.IO connection
   - Setup UI controls (name input, start button, mic toggle, screen share)

2. **Session Start**
   - User enters name → `socket.emit('candidate_info', {name})`
   - Click "Start Session" → `startWebcam()` function
   - Request camera permissions → `navigator.mediaDevices.getUserMedia()`

3. **Media Capture & Streaming**
   - **Webcam Stream:** 640x480 @ 8 FPS → Canvas → Base64 → Socket emit
   - **Screen Share:** Display media @ 3 FPS → Canvas → Base64 → Socket emit  
   - **Audio Stream:** 16kHz PCM → ScriptProcessor → Base64 → Socket emit

**Tech Stack:**
- **Frontend:** HTML5 Canvas API, WebRTC getUserMedia/displayMedia
- **Audio Processing:** Web Audio API (AudioContext, ScriptProcessor)
- **Data Format:** Base64 encoded images/audio
- **Transport:** Socket.IO WebSocket

### **3. SERVER PROCESSING FLOW**
```
Frame Reception → MediaPipe Analysis → Behavior Metrics → AI Processing → HR Dashboard
```

**Detailed Steps:**
1. **Frame Reception** (`server.py:on_frame()`)
   - Receive Base64 image from candidate
   - Decode to OpenCV BGR format
   - Determine frame type (webcam/screen)

2. **AI Analysis Pipeline** (for webcam frames)
   - **MediaPipe Holistic Processing:**
     - Pose landmarks (33 points)
     - Face landmarks (468 points) 
     - Hand landmarks (21 points each hand)
   - **Behavior Analysis** (`behavior_analyzer.py`):
     - Posture estimation (upright/leaning/slouching)
     - Head pose (roll/yaw/pitch)
     - Eye state (open/closed ratio)
     - Mouth state (open/closed ratio)
     - Fidgeting detection (motion history)
     - Hand-to-face proximity

3. **Audio Analysis** (`server.py:on_audio()`)
   - Decode Base64 PCM audio
   - RMS loudness calculation
   - Pitch estimation via autocorrelation
   - Nervousness/cheating heuristics

4. **Metrics Combination**
   - Visual confidence score
   - Cheating risk meter
   - Nervousness meter
   - Real-time dashboard updates

**Tech Stack:**
- **AI/ML:** MediaPipe Holistic, OpenCV, NumPy
- **Audio Processing:** NumPy, SciPy (autocorrelation)
- **Computer Vision:** OpenCV (image processing)
- **Real-time:** Socket.IO, threading

### **4. HR DASHBOARD FLOW**
```
Dashboard Load → Real-time Updates → AI Metrics Display → Follow-up Questions
```

**Detailed Steps:**
1. **Dashboard Initialization**
   - Load hr.html template
   - Connect to Socket.IO
   - Initialize UI components

2. **Real-time Data Reception**
   - Receive candidate info → Update name display
   - Receive webcam frames → Display video feed
   - Receive screen frames → Display screen share
   - Receive behavior metrics → Update analysis panel
   - Receive audio metrics → Update audio panel
   - Receive combined meters → Update confidence/cheating/nervous meters

3. **Smart Follow-up Questions**
   - Questions categorized: Behavioral, Technical, Scenario-based, Clarification
   - Dynamic highlighting based on AI analysis
   - Click-to-copy functionality
   - Real-time question relevance

**Tech Stack:**
- **Frontend:** HTML5, CSS3, JavaScript ES6+
- **Real-time:** Socket.IO client
- **UI/UX:** Custom CSS with dark theme, responsive design
- **Interactivity:** Event listeners, clipboard API

## 🛠️ **DETAILED TECH STACK BREAKDOWN**

### **BACKEND TECHNOLOGIES**

#### **Core Framework**
- **Flask 3.0+** - Python web framework
- **Flask-SocketIO 5.3+** - WebSocket support
- **Eventlet 0.36+** - Async networking

#### **AI/Computer Vision**
- **MediaPipe 0.10+** - Google's ML framework
  - Holistic model (pose + face + hands)
  - Real-time landmark detection
  - 33 pose landmarks, 468 face landmarks, 21 hand landmarks each
- **OpenCV 4.8+** - Computer vision library
  - Image processing and manipulation
  - Video capture and encoding
  - BGR/RGB color space conversion
- **NumPy 1.24+** - Numerical computing
  - Array operations for landmarks
  - Mathematical calculations
  - Audio signal processing

#### **Audio Processing**
- **SciPy 1.10+** - Scientific computing
  - Signal processing algorithms
  - Autocorrelation for pitch detection
- **Custom Audio Analysis:**
  - RMS loudness calculation
  - Pitch estimation via autocorrelation
  - Nervousness/cheating heuristics

#### **Screen Recording**
- **MSS 9.0+** - Multi-screen screenshot library
  - Cross-platform screen capture
  - Monitor selection and recording
- **OpenCV VideoWriter** - MP4 video encoding

### **FRONTEND TECHNOLOGIES**

#### **Core Web Technologies**
- **HTML5** - Semantic markup
- **CSS3** - Styling and animations
  - Flexbox/Grid layouts
  - Dark theme design
  - Responsive design
  - CSS transitions and transforms
- **JavaScript ES6+** - Client-side logic
  - Async/await for media APIs
  - Canvas API for image processing
  - Web Audio API for audio processing

#### **Web APIs**
- **getUserMedia()** - Camera/microphone access
- **getDisplayMedia()** - Screen sharing
- **Web Audio API** - Real-time audio processing
- **Canvas API** - Image capture and manipulation
- **Clipboard API** - Copy functionality

#### **Real-time Communication**
- **Socket.IO 4.7+** - WebSocket library
  - Real-time bidirectional communication
  - Automatic reconnection
  - Event-based messaging

### **DATA FLOW TECHNOLOGIES**

#### **Image Processing Pipeline**
```
Webcam → Canvas → Base64 → Socket.IO → Server → OpenCV → MediaPipe → Metrics → Dashboard
```

#### **Audio Processing Pipeline**
```
Microphone → Web Audio API → PCM → Base64 → Socket.IO → Server → NumPy → Analysis → Dashboard
```

#### **Screen Recording Pipeline**
```
Screen → getDisplayMedia() → Canvas → Base64 → Socket.IO → Server → OpenCV → MP4 File
```

## 📊 **PERFORMANCE CHARACTERISTICS**

### **Frame Rates**
- **Webcam:** 8 FPS (optimized for analysis)
- **Screen Share:** 3 FPS (reduced bandwidth)
- **Audio:** Real-time processing (16kHz)

### **Data Sizes**
- **Images:** ~50-100KB per frame (JPEG compressed)
- **Audio:** ~4KB per chunk (PCM 16-bit)
- **Metrics:** <1KB per update (JSON)

### **Latency**
- **Analysis:** ~50-100ms per frame
- **Network:** <100ms (local network)
- **Total:** ~200-300ms end-to-end

## 🔧 **DEPLOYMENT & RUNTIME**

### **Server Requirements**
- **Python 3.8+**
- **Camera/Microphone access**
- **Sufficient RAM for MediaPipe**
- **Network connectivity**

### **Client Requirements**
- **Modern browser with WebRTC support**
- **Camera and microphone permissions**
- **JavaScript enabled**
- **Stable internet connection**

### **File Structure**
```
├── server.py              # Main Flask server
├── behavior_analyzer.py   # AI analysis engine
├── run_webcam.py         # Standalone webcam test
├── screen_recorder.py    # Screen recording utility
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── index.html       # Landing page
│   ├── candidate.html   # Candidate interface
│   └── hr.html         # HR dashboard
└── captures/            # Recorded screen sessions
```

## 🚀 **KEY FEATURES IMPLEMENTED**

1. **Real-time AI Analysis** - Live behavior monitoring
2. **Multi-modal Input** - Webcam + Screen + Audio
3. **Smart Follow-up Questions** - Context-aware interview questions
4. **Professional UI** - Dark theme, responsive design
5. **Screen Recording** - Automatic session recording
6. **Cross-platform** - Works on Windows/Mac/Linux
7. **Scalable Architecture** - Modular, maintainable code

This system provides a comprehensive AI-powered interview analysis platform with real-time behavioral insights and intelligent follow-up question suggestions for HR professionals.
