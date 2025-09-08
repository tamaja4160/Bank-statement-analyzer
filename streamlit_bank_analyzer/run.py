#!/usr/bin/env python3
"""
Run script for the Bank Statement Analyzer Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import cv2
        import pytesseract
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        import pytesseract
        # Set the tesseract path from config before testing
        try:
            import sys
            import os
            parent_dir = os.path.join(os.path.dirname(__file__), '..')
            sys.path.insert(0, parent_dir)
            from simple_image_reader.config import TESSERACT_CMD
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR found: {version}")
            return True
        except ImportError:
            # Fallback to default path
            try:
                version = pytesseract.get_tesseract_version()
                print(f"✅ Tesseract OCR found: {version}")
                return True
            except Exception as e:
                print(f"❌ Tesseract OCR not found: {e}")
                print("Please install Tesseract OCR:")
                print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
                print("  macOS: brew install tesseract")
                print("  Linux: sudo apt-get install tesseract-ocr")
                return False
    except ImportError as e:
        print(f"❌ pytesseract not installed: {e}")
        return False

def main():
    """Main function to run the Streamlit app."""
    print("🏦 Bank Statement Analyzer")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the streamlit_bank_analyzer directory")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check Tesseract
    if not check_tesseract():
        sys.exit(1)

    print("\n🚀 Starting Streamlit application...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, visit: http://localhost:8501")
    print("\n" + "=" * 50)

    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
