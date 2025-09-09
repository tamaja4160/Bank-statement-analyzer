#!/usr/bin/env python3
"""
Main runner script for the Bank Statement Suite.

This script launches the primary Streamlit application for bank statement analysis,
which provides an integrated interface to all suite features.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit bank analyzer application."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    streamlit_dir = script_dir / "streamlit_bank_analyzer"

    if not streamlit_dir.exists():
        print("❌ Error: streamlit_bank_analyzer directory not found!")
        print("Please ensure you're running from the correct directory.")
        sys.exit(1)

    print("🚀 Starting Bank Statement Suite...")
    print("📊 Launching Streamlit application...")

    # Change to the streamlit directory
    os.chdir(streamlit_dir)

    # Launch Streamlit
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")

if __name__ == "__main__":
    main()
