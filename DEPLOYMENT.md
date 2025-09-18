# Render Deployment Guide

## Prerequisites
1. GitHub repository with your code
2. Render account (free tier available)

## Deployment Steps

### 1. Prepare Your Repository
Make sure these files are in your repository root:
- `server.py` (main application file)
- `requirements.txt` (Python dependencies)
- `runtime.txt` (Python version)
- `render.yaml` (Render configuration)
- `Procfile` (alternative start command)
- `.renderignore` (files to exclude)

### 2. Deploy on Render

#### Option A: Using render.yaml (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file
5. Click "Create Web Service"

#### Option B: Manual Configuration
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: ai-interview-analysis
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Plan**: Free

### 3. Environment Variables
Add these environment variables in Render dashboard:
- `PORT`: 8000 (Render will set this automatically)
- `SECRET_KEY`: Generate a random secret key

### 4. Important Notes

#### Dependencies
- Using `opencv-python-headless` instead of `opencv-python` for better compatibility
- Some heavy dependencies (MediaPipe, YOLO) may take time to install
- Free tier has memory limits that might affect performance

#### Limitations on Free Tier
- **Memory**: 512MB RAM limit
- **CPU**: Shared resources
- **Build Time**: 90 minutes max
- **Sleep**: Service sleeps after 15 minutes of inactivity

#### Potential Issues
1. **MediaPipe**: May require additional system dependencies
2. **YOLO Model**: Large model file may cause memory issues
3. **OpenCV**: Use headless version for server environments

### 5. Troubleshooting

#### Build Failures
- Check the build logs in Render dashboard
- Ensure all dependencies are compatible
- Consider removing heavy dependencies for initial deployment

#### Runtime Errors
- Check the service logs
- Verify environment variables are set
- Test locally first

#### Memory Issues
- Monitor memory usage in Render dashboard
- Consider upgrading to paid plan if needed
- Optimize code to use less memory

### 6. Alternative Deployment Options

If Render doesn't work due to dependency issues:

#### Heroku
- Similar to Render but different buildpack
- May handle heavy dependencies better

#### Railway
- Good for Python applications
- Automatic deployments from GitHub

#### DigitalOcean App Platform
- More control over the environment
- Better for complex applications

### 7. Testing Your Deployment

Once deployed:
1. Visit your Render URL
2. Test the HR dashboard: `https://your-app.onrender.com/hr`
3. Test the candidate interface: `https://your-app.onrender.com/candidate`
4. Verify PDF generation works

### 8. Monitoring

- Check Render dashboard for logs
- Monitor memory and CPU usage
- Set up alerts for service downtime
