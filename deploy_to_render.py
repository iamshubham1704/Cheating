#!/usr/bin/env python3
"""
Script to prepare and deploy to Render
"""

import os
import subprocess
import sys

def check_git_status():
    """Check if we're in a git repository and if there are changes"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository. Please initialize git first:")
            print("   git init")
            print("   git add .")
            print("   git commit -m 'Initial commit'")
            return False
        
        if result.stdout.strip():
            print("📝 Changes detected:")
            print(result.stdout)
            return True
        else:
            print("✅ No changes to commit")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install git first.")
        return False

def commit_and_push():
    """Commit changes and push to GitHub"""
    try:
        print("🔄 Adding files...")
        subprocess.run(['git', 'add', '.'], check=True)
        
        print("💾 Committing changes...")
        subprocess.run(['git', 'commit', '-m', 'Fix Render deployment - remove heavy dependencies'], check=True)
        
        print("🚀 Pushing to GitHub...")
        subprocess.run(['git', 'push'], check=True)
        
        print("✅ Successfully pushed to GitHub!")
        print("\n🎉 Next steps:")
        print("1. Go to https://dashboard.render.com")
        print("2. Create new web service")
        print("3. Connect your GitHub repository")
        print("4. Deploy!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git not found. Please install git first.")
        return False

def main():
    print("🚀 Render Deployment Helper")
    print("=" * 40)
    
    # Check if we're in a git repo
    if not check_git_status():
        return
    
    # Ask user if they want to proceed
    response = input("\n🤔 Do you want to commit and push these changes? (y/n): ")
    if response.lower() != 'y':
        print("❌ Deployment cancelled")
        return
    
    # Commit and push
    if commit_and_push():
        print("\n🎉 Deployment preparation complete!")
    else:
        print("\n❌ Deployment preparation failed")

if __name__ == "__main__":
    main()
