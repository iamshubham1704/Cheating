# Camera Troubleshooting Guide for Render Deployment

## 🎯 **Common Camera Issues on Render**

Your application is successfully deployed on Render, but camera access might not work due to several factors. Here's a comprehensive guide to resolve these issues.

## 🔧 **Quick Fixes**

### 1. **HTTPS Requirement** ✅
- **Problem**: Camera access requires HTTPS in production
- **Solution**: Your Render app automatically provides HTTPS
- **URL Format**: `https://your-app-name.onrender.com`

### 2. **Browser Permissions** ✅
- **Problem**: Browser blocks camera access
- **Solution**: 
  1. Click the camera icon in your browser's address bar
  2. Select "Allow" for camera and microphone access
  3. Refresh the page and try again

### 3. **Browser Compatibility** ✅
- **Recommended Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile**: Works on mobile browsers with camera support

## 🚨 **Error Messages & Solutions**

### "Camera permission denied"
- **Cause**: User denied camera access
- **Solution**: 
  1. Click the camera icon in address bar
  2. Change to "Allow"
  3. Refresh page

### "Camera access not supported"
- **Cause**: Browser doesn't support getUserMedia
- **Solution**: Use a modern browser (Chrome, Firefox, Safari, Edge)

### "No camera found"
- **Cause**: No camera connected or detected
- **Solution**: 
  1. Connect a camera
  2. Check camera is not being used by another app
  3. Restart browser

### "Camera is being used by another application"
- **Cause**: Another app is using the camera
- **Solution**: 
  1. Close other video apps (Zoom, Teams, etc.)
  2. Try again

### "Camera access blocked by security policy"
- **Cause**: Browser security settings
- **Solution**: 
  1. Ensure you're using HTTPS
  2. Check browser security settings
  3. Try incognito/private mode

## 🔍 **Step-by-Step Testing**

### Test 1: Basic Access
1. Go to `https://your-app-name.onrender.com/candidate`
2. Enter your name
3. Click "Start Session"
4. Allow camera access when prompted

### Test 2: Permission Check
1. If camera doesn't work, check browser permissions:
   - **Chrome**: `chrome://settings/content/camera`
   - **Firefox**: `about:preferences#privacy` → Permissions → Camera
   - **Safari**: Safari → Preferences → Websites → Camera

### Test 3: Browser Console
1. Open Developer Tools (F12)
2. Check Console for error messages
3. Look for specific error names (NotAllowedError, NotFoundError, etc.)

## 🛠️ **Advanced Troubleshooting**

### For Developers
1. **Check HTTPS**: Ensure your Render URL uses `https://`
2. **Test Locally**: Test camera access on `localhost` first
3. **Browser DevTools**: Check for WebRTC errors in console
4. **Network**: Ensure stable internet connection

### For Users
1. **Clear Browser Data**: Clear cookies and site data
2. **Disable Extensions**: Try with extensions disabled
3. **Different Browser**: Test with different browser
4. **Restart Browser**: Close and reopen browser

## 📱 **Mobile Considerations**

### iOS Safari
- Camera access works but may have limitations
- User must explicitly allow permissions
- Some features may not work on older iOS versions

### Android Chrome
- Generally works well
- May require additional permissions
- Check Android app permissions

## 🔄 **Updated Features**

The application now includes:

1. **Better Error Handling**: Specific error messages for different camera issues
2. **HTTPS Detection**: Automatic detection of secure connection
3. **Fallback Options**: Basic camera settings if advanced features fail
4. **User Guidance**: Clear instructions for resolving issues
5. **Permission Retry**: Automatic retry with basic settings

## 🎯 **Expected Behavior**

### Successful Camera Access
- Camera preview appears in the webcam section
- Status shows "Camera active - streaming video"
- Video frames are sent to the server

### Failed Camera Access
- Clear error message displayed
- Instructions shown for resolution
- Option to retry with basic settings

## 🆘 **Still Having Issues?**

If camera access still doesn't work:

1. **Check Render Logs**: Look at your Render service logs for server-side errors
2. **Test Different Devices**: Try on different computers/phones
3. **Browser Version**: Ensure you're using a recent browser version
4. **Network**: Check if your network blocks camera access

## 📞 **Support**

For additional help:
1. Check browser console for specific error messages
2. Test on different browsers and devices
3. Verify HTTPS is working (green lock icon in address bar)
4. Ensure camera is not being used by other applications

---

**Note**: Camera access is a browser security feature and requires user permission. The application will guide users through the permission process and provide clear error messages if access is denied.
