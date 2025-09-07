#!/usr/bin/env python3
"""
Detailed debug script to understand why transactions aren't being parsed.
"""

import re
from bank_statement_parser import BankStatementParser

def debug_parser_step_by_step():
    """Debug the parser step by step."""
    
    # Initialize parser
    parser = BankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
    
    # Get OCR text from statement_1.png
    image_path = "bank statement generator/bank_statements/statement_1.png"
    text = parser.extract_text_from_image(image_path)
    statement_id = parser.extract_statement_id(text, image_path)
    
    print(f"Statement ID: {statement_id}")
    print(f"OCR Text:\n{text}")
    print("\n=== STEP BY STEP PARSING ===")
    
    lines = text.split('\n')
    transaction_pattern = r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+?)\s+(\d+(?:[,\.]\d{1,2})?)[,.]?'
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        print(f"\nLine {line_num}: '{line}'")
        
        # Try to match transaction pattern
        match = re.search(transaction_pattern, line)
        if match:
            beleg_date = match.group(1)
            valuta_date = match.group(2)
            description = match.group(3).strip()
            amount_str = match.group(4)
            
            print(f"  ✅ Regex match found:")
            print(f"    Beleg date: '{beleg_date}'")
            print(f"    Valuta date: '{valuta_date}'")
            print(f"    Description: '{description}'")
            print(f"    Amount string: '{amount_str}'")
            
            # Test date parsing
            parsed_date = parser.parse_date(beleg_date)
            print(f"    Parsed date: {parsed_date}")
            
            # Test amount parsing
            parsed_amount = parser.parse_amount(amount_str)
            print(f"    Parsed amount: {parsed_amount}")
            
            if parsed_date and parsed_amount is not None:
                partner = parser.identify_partner(description)
                category = parser.classify_category(partner, description)
                
                print(f"    ✅ Transaction would be created:")
                print(f"      Partner: {partner}")
                print(f"      Category: {category}")
            else:
                print(f"    ❌ Transaction NOT created - date or amount parsing failed")
                if not parsed_date:
                    print(f"      Date parsing failed for: '{beleg_date}'")
                if parsed_amount is None:
                    print(f"      Amount parsing failed for: '{amount_str}'")
        else:
            print(f"  ❌ No regex match")

if __name__ == "__main__":
    debug_parser_step_by_step()
