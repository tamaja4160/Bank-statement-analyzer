#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for the Bank Statement Analyzer.
"""

import sys
import os

def test_imports():
    """Test all required imports for the application."""
    print("🧪 Testing imports for Bank Statement Analyzer")
    print("=" * 60)

    # Test basic Python packages
    try:
        import pandas as pd
        print(f"✅ pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")

    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")

    # Test Streamlit
    try:
        import streamlit as st
        print(f"✅ streamlit: {st.__version__}")
    except ImportError as e:
        print(f"❌ streamlit: {e}")

    # Test OpenCV and OCR
    try:
        import cv2
        print(f"✅ opencv-python: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ opencv-python: {e}")

    try:
        import pytesseract
        # Set the tesseract path from config before testing
        try:
            # Add parent directory to path first
            parent_dir = os.path.join(os.path.dirname(__file__), '..')
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from simple_image_reader.config import TESSERACT_CMD
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            version = pytesseract.get_tesseract_version()
            print(f"✅ pytesseract: {version}")
        except ImportError:
            # Fallback to default path
            try:
                version = pytesseract.get_tesseract_version()
                print(f"✅ pytesseract: {version}")
            except Exception as e:
                print(f"⚠️ pytesseract: Installed but Tesseract not found - {e}")
    except ImportError as e:
        print(f"❌ pytesseract: {e}")

    # Test Excel processing
    try:
        import openpyxl
        print(f"✅ openpyxl: {openpyxl.__version__}")
    except ImportError as e:
        print(f"❌ openpyxl: {e}")

    try:
        import excel2img
        print("✅ excel2img: Available")
    except ImportError as e:
        print(f"❌ excel2img: {e}")

    # Test Faker
    try:
        from faker import Faker
        print("✅ faker: Available")
    except ImportError as e:
        print(f"❌ faker: {e}")

    # Test fuzzy matching
    try:
        from fuzzywuzzy import fuzz
        print("✅ fuzzywuzzy: Available")
    except ImportError as e:
        print(f"❌ fuzzywuzzy: {e}")

    # Test our custom modules
    print("\n📦 Testing custom modules:")

    try:
        from statement_generator import generate_statements_for_name
        print("✅ statement_generator: OK")
    except ImportError as e:
        print(f"❌ statement_generator: {e}")

    try:
        from payment_analyzer import analyze_recurring_payments
        print("✅ payment_analyzer: OK")
    except ImportError as e:
        print(f"❌ payment_analyzer: {e}")

    try:
        from session_manager import SessionManager
        print("✅ session_manager: OK")
    except ImportError as e:
        print(f"❌ session_manager: {e}")

    try:
        from display_helpers import display_statement_with_extractions
        print("✅ display_helpers: OK")
    except ImportError as e:
        print(f"❌ display_helpers: {e}")

    # Test simple_image_reader integration
    print("\n🔗 Testing integration with simple_image_reader:")

    # Add parent directory to path for testing
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from simple_image_reader.simple_image_reader import SimpleImageReader
        print("✅ simple_image_reader: OK")
    except ImportError as e:
        print(f"❌ simple_image_reader: {e}")

    print("\n" + "=" * 60)
    print("🎉 Import test completed!")
    print("\n💡 Next steps:")
    print("1. If any imports failed, run: pip install -r requirements.txt")
    print("2. Install Tesseract OCR if pytesseract shows warnings")
    print("3. Run the app with: python run.py")

if __name__ == "__main__":
    test_imports()
