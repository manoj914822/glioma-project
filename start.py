#!/usr/bin/env python3
"""
Startup script for GLIOMA Brain Tumor Analysis Application
This script checks dependencies and starts the application
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print("   Please upgrade Python and try again")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('flask', 'flask'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('tensorflow', None),
        ('pillow', 'PIL')
    ]

    missing_packages = []

    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name} is installed")
        except ImportError:
            missing_packages.append(display_name)
            print(f"❌ {display_name} is missing")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'static',
        'static/uploads',
        'static/processed',
        'templates',
        'models',
        'instance'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ All directories created")

def start_application():
    """Start the Flask application"""
    try:
        print("\n🚀 Starting GLIOMA Brain Tumor Analysis Application...")
        print("=" * 60)
        print("🌐 Application will be available at: http://127.0.0.1:5000")
        print("📱 Open your web browser and navigate to the above URL")
        print("🛑 Press Ctrl+C to stop the application")
        print("=" * 60)
        
        # Import and run the app
        from app import app
        app.run(debug=True, host='127.0.0.1', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing application: {e}")
        print("   Make sure app.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main startup function"""
    print("🧠 GLIOMA Brain Tumor Analysis Application")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or run the setup script:")
        print("   python setup.py")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Start application
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        print("Thank you for using GLIOMA Brain Tumor Analysis!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
