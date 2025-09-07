#!/usr/bin/env python3
"""
Debug script to test regex patterns against actual OCR output.
"""

import re
import os
import cv2
import numpy as np
import pytesseract
from bank_statement_parser import BankStatementParser

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def test_regex_patterns():
    """Test different regex patterns against OCR output."""
    
    # Initialize parser to get OCR text
    parser = BankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
    
    # Get OCR text from statement_1.png
    image_path = "bank statement generator/bank_statements/statement_1.png"
    text = parser.extract_text_from_image(image_path)
    
    print("=== OCR TEXT ===")
    print(text)
    print("\n=== TESTING REGEX PATTERNS ===")
    
    # Test different patterns
    patterns = [
        # Current pattern
        r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+?)\s+(\d+[,\.]\d{2})[,.]',
        
        # More flexible patterns
        r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+?)\s+(\d+[,\.]\d{2})',
        r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+)\s+(\d+[,\.]\d{2})[,.]',
        r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+)\s+(\d+[,\.]\d{2})',
        
        # Even more flexible
        r'(\d{2}\.\d{2}),?\s+(\d{2}\.\d{2}),?\s+(.+?)\s+(\d+[,\.]\d{2})',
        r'(\d{2}\.\d{2})[,.]?\s+(\d{2}\.\d{2})[,.]?\s+(.+?)\s+(\d+[,\.]\d{2})',
    ]
    
    lines = text.split('\n')
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\nPattern {i}: {pattern}")
        matches_found = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            matches = re.findall(pattern, line)
            if matches:
                matches_found += len(matches)
                print(f"  Line {line_num}: '{line}'")
                for match in matches:
                    print(f"    Match: {match}")
        
        print(f"  Total matches: {matches_found}")
    
    print("\n=== MANUAL LINE ANALYSIS ===")
    # Let's manually check the transaction lines we know exist
    transaction_lines = [
        "07.06, 08.06, BEITRAG Allianz SEK-71445329 56.28.",
        "08.06, 9.06, BURGER KING Vilsbiburg was",
        "11.06, 12.06, RECHNUNG VODAFONE GMBH 2925470 3963.",
        "12.06, 13.06, RARTENZAHLUNG JET TANKSTELLE 65.62,",
        "15.06, 16.06, RARTENZAHLUNG JET TANKSTELLE 52.18."
    ]
    
    # Test our current pattern against these known lines
    current_pattern = r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+?)\s+(\d+[,\.]\d{2})[,.]'
    
    for line in transaction_lines:
        print(f"\nTesting line: '{line}'")
        match = re.search(current_pattern, line)
        if match:
            print(f"  ✅ Match: {match.groups()}")
        else:
            print(f"  ❌ No match")
            
            # Try to understand why it doesn't match
            # Test parts of the pattern
            date_pattern = r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),'
            desc_amount_pattern = r'(.+?)\s+(\d+[,\.]\d{2})[,.]'
            
            date_match = re.search(date_pattern, line)
            if date_match:
                print(f"    Date part matches: {date_match.groups()}")
                remaining = line[date_match.end():].strip()
                print(f"    Remaining text: '{remaining}'")
                
                amount_match = re.search(desc_amount_pattern, remaining)
                if amount_match:
                    print(f"    Amount part matches: {amount_match.groups()}")
                else:
                    print(f"    Amount part doesn't match")
            else:
                print(f"    Date part doesn't match")

if __name__ == "__main__":
    test_regex_patterns()
