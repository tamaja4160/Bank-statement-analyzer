"""Debug version to analyze OCR extraction issues."""

import sys
from pathlib import Path
from simple_image_reader import SimpleImageReader

def debug_single_image(image_path: str):
    """Debug a single image to see OCR text and extraction results."""
    reader = SimpleImageReader()
    
    image_path = Path(image_path)
    result = reader.process_single_image(image_path)
    
    print(f"=== DEBUG ANALYSIS FOR {image_path.name} ===")
    print(f"OCR Confidence: {result['ocr_confidence']:.1f}%")
    print(f"Transactions found: {result['extracted_count']}")
    print()
    
    print("=== RAW OCR TEXT ===")
    print(result['raw_ocr_text'])
    print()
    
    print("=== EXTRACTED TRANSACTIONS ===")
    for i, trans in enumerate(result['transactions'], 1):
        print(f"{i}. {trans['raw_line']}")
    print()
    
    # Show OCR text lines for analysis
    print("=== OCR TEXT LINES ===")
    lines = [line.strip() for line in result['raw_ocr_text'].split('\n') if line.strip()]
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}: {line}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_single_image(sys.argv[1])
    else:
        print("Usage: python debug_extractor.py <image_path>")
