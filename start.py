#!/usr/bin/env python3
"""
Startup script for Render deployment
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the deployment server
try:
    from server_deploy import app, socketio
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting AI Interview Analysis Server on port {port}")
        print("Using simplified deployment version...")
        
        # Run the server
        socketio.run(
            app, 
            host="0.0.0.0", 
            port=port, 
            debug=False, 
            allow_unsafe_werkzeug=True
        )
        
except ImportError as e:
    print(f"Error importing server_deploy: {e}")
    print("Falling back to main server...")
    
    try:
        from server import app, socketio
        
        if __name__ == "__main__":
            port = int(os.environ.get("PORT", 8000))
            print(f"Starting AI Interview Analysis Server on port {port}")
            print("Using main server version...")
            
            # Run the server
            socketio.run(
                app, 
                host="0.0.0.0", 
                port=port, 
                debug=False, 
                allow_unsafe_werkzeug=True
            )
    except ImportError as e2:
        print(f"Error importing main server: {e2}")
        print("No server module found!")
        sys.exit(1)
