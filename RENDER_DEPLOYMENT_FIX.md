# Render Deployment Fix

## Problem
Render is using Python 3.13.4, but MediaPipe doesn't support Python 3.13 yet. Also, it's trying to install heavy dependencies that cause build failures.

## Solution Applied

### 1. Updated requirements.txt
Removed heavy dependencies that don't work on Render free tier:
- ❌ mediapipe (not compatible with Python 3.13)
- ❌ ultralytics (too heavy)
- ❌ torch (too heavy)
- ❌ scipy (causing build issues)
- ❌ mss (not needed for basic functionality)

### 2. Kept Essential Dependencies
- ✅ Flask (web framework)
- ✅ Flask-SocketIO (real-time communication)
- ✅ eventlet (async support)
- ✅ reportlab (PDF generation)
- ✅ Pillow (image processing)
- ✅ numpy (basic math)
- ✅ opencv-python-headless (video processing)

### 3. Created Simplified Server
- `server_deploy.py` - Works without heavy AI dependencies
- Maintains core functionality: video streaming, PDF reports, leave room features
- Graceful error handling for missing dependencies

## Deployment Steps

### Option 1: Automatic (Recommended)
1. Push all changes to your GitHub repository
2. Go to Render dashboard
3. Create new web service
4. Connect your GitHub repository
5. Render will automatically detect `render.yaml` and use the correct settings

### Option 2: Manual Configuration
If automatic detection doesn't work:
1. Go to Render dashboard
2. Create new web service
3. Connect your GitHub repository
4. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server_deploy.py`
   - **Environment**: Python 3
   - **Python Version**: 3.11.9 (specified in runtime.txt)

## What Works After Deployment

### ✅ Core Features
- Video streaming (webcam and screen share)
- Real-time communication
- Leave interview room functionality
- PDF report generation
- Basic audio processing

### ❌ Not Available (Due to Dependencies)
- AI behavior analysis (MediaPipe)
- Object detection (YOLO)
- Advanced posture analysis
- Cheating detection algorithms

## Testing Your Deployment

1. Visit your Render URL
2. Test HR dashboard: `https://your-app.onrender.com/hr`
3. Test candidate interface: `https://your-app.onrender.com/candidate`
4. Test PDF generation after ending an interview

## If You Need Full AI Features

Consider these alternatives:
1. **Upgrade Render Plan** - More memory and resources
2. **Railway** - Better for heavy Python applications
3. **DigitalOcean App Platform** - More control over environment
4. **Heroku** - Good for Python apps with add-ons

## Troubleshooting

### Build Still Fails
- Check that all files are pushed to GitHub
- Verify `runtime.txt` specifies Python 3.11.9
- Check Render logs for specific error messages

### App Starts But Doesn't Work
- Check service logs in Render dashboard
- Verify environment variables are set
- Test locally first with `python server_deploy.py`

### Need to Add AI Features Back
- Upgrade to Render paid plan
- Or use alternative deployment platform
- Or modify code to use lighter AI libraries
