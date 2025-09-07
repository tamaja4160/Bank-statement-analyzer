#!/usr/bin/env python3
"""
Test script to verify the updated bank statement parser works on sample images.
"""

import os
import sys
from bank_statement_parser import BankStatementParser

def test_parser_on_samples():
    """Test the parser on a few sample images."""
    
    # Configuration
    GROUND_TRUTH_PATH = "bank statement generator/bank_statements/ground_truth.csv"
    IMAGES_FOLDER = "bank statement generator/bank_statements"
    
    # Check if required files exist
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Error: Ground truth file not found: {GROUND_TRUTH_PATH}")
        return
    
    if not os.path.exists(IMAGES_FOLDER):
        print(f"❌ Error: Images folder not found: {IMAGES_FOLDER}")
        return
    
    print("🧪 Testing Bank Statement Parser on Sample Images...")
    
    # Initialize parser
    parser = BankStatementParser(GROUND_TRUTH_PATH)
    
    # Test on first few images
    sample_images = ["statement_1.png", "statement_2.png", "statement_10.png"]
    
    for image_name in sample_images:
        image_path = os.path.join(IMAGES_FOLDER, image_name)
        if os.path.exists(image_path):
            print(f"\n📄 Testing {image_name}...")
            
            # Parse single statement
            transactions = parser.parse_single_statement(image_path)
            
            print(f"✅ Found {len(transactions)} transactions:")
            for i, transaction in enumerate(transactions, 1):
                print(f"  {i}. Date: {transaction['date']}")
                print(f"     Description: {transaction['description']}")
                print(f"     Amount: {transaction['amount']:.2f}")
                print(f"     Partner: {transaction['identified_partner']}")
                print(f"     Category: {transaction['category']}")
                print()
        else:
            print(f"❌ Image not found: {image_path}")
    
    print("🏁 Sample testing completed!")

if __name__ == "__main__":
    test_parser_on_samples()
