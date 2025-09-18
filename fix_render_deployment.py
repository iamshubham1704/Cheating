#!/usr/bin/env python3
"""
Script to fix Render deployment issues
"""

import os
import sys

def create_deployment_files():
    """Create necessary files for Render deployment"""
    
    # Ensure runtime.txt exists and is correct
    if not os.path.exists('runtime.txt'):
        with open('runtime.txt', 'w') as f:
            f.write('python-3.11.9\n')
        print("✓ Created runtime.txt")
    else:
        print("✓ runtime.txt already exists")
    
    # Create .renderignore if it doesn't exist
    if not os.path.exists('.renderignore'):
        ignore_content = """__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
captures/
*.mp4
*.avi
*.mov
.DS_Store
Thumbs.db
yolov8n.pt
"""
        with open('.renderignore', 'w') as f:
            f.write(ignore_content)
        print("✓ Created .renderignore")
    else:
        print("✓ .renderignore already exists")
    
    # Create Procfile if it doesn't exist
    if not os.path.exists('Procfile'):
        with open('Procfile', 'w') as f:
            f.write('web: python server_deploy.py\n')
        print("✓ Created Procfile")
    else:
        print("✓ Procfile already exists")
    
    print("\n🎉 Deployment files are ready!")
    print("\nNext steps:")
    print("1. Push these changes to your GitHub repository")
    print("2. Go to Render dashboard and create a new web service")
    print("3. Connect your GitHub repository")
    print("4. Use the following settings:")
    print("   - Build Command: pip install -r requirements_deploy.txt")
    print("   - Start Command: python server_deploy.py")
    print("   - Environment: Python 3")
    print("5. Deploy!")

if __name__ == "__main__":
    create_deployment_files()
