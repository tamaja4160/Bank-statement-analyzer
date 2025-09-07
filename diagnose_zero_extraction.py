import os
import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple
import re
from enhanced_bank_statement_parser import EnhancedBankStatementParser

class ZeroExtractionDiagnostic:
    def __init__(self):
        self.parser = EnhancedBankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
        
    def analyze_statement(self, image_path: str, statement_id: int) -> Dict:
        """Analyze a single statement and return detailed diagnostic info"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {image_path}")
        print(f"{'='*60}")
        
        # Get OCR text with confidence
        image_variants = self.parser.advanced_preprocess_image(image_path)
        ocr_text = self.parser.extract_text_with_multiple_configs(image_variants)
        
        # Try to extract transactions
        transactions = self.parser.enhanced_parse_transactions_from_text(ocr_text, statement_id)
        
        # Analyze the text for potential transaction patterns
        potential_patterns = self.find_potential_transaction_patterns(ocr_text)
        
        result = {
            'statement_id': statement_id,
            'image_path': image_path,
            'ocr_text_length': len(ocr_text),
            'extracted_transactions': len(transactions),
            'transactions': transactions,
            'ocr_text': ocr_text,
            'potential_patterns': potential_patterns
        }
        
        print(f"OCR Text Length: {len(ocr_text)} characters")
        print(f"Extracted Transactions: {len(transactions)}")
        print(f"Potential Patterns Found: {len(potential_patterns)}")
        
        print(f"\nOCR TEXT (first 500 chars):")
        print("-" * 40)
        print(ocr_text[:500])
        print("-" * 40)
        
        if transactions:
            print(f"\nEXTRACTED TRANSACTIONS:")
            for i, trans in enumerate(transactions, 1):
                print(f"  {i}. Date: {trans.get('date', 'N/A')}, Amount: {trans.get('amount', 'N/A')}, Partner: {trans.get('partner', 'N/A')}")
        else:
            print(f"\nNO TRANSACTIONS EXTRACTED!")
            
        if potential_patterns:
            print(f"\nPOTENTIAL TRANSACTION PATTERNS:")
            for i, pattern in enumerate(potential_patterns, 1):
                print(f"  {i}. {pattern}")
        
        return result
    
    def find_potential_transaction_patterns(self, text: str) -> List[str]:
        """Find potential transaction patterns in the text that might be missed"""
        patterns = []
        
        # Look for date-like patterns
        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # DD.MM.YYYY or DD.MM.YY
            r'\d{1,2}\.\d{1,2}',           # DD.MM
            r'\d{1,2}/\d{1,2}/\d{2,4}',    # DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',    # DD-MM-YYYY
        ]
        
        # Look for amount-like patterns
        amount_patterns = [
            r'\d+[,\.]\d{2}',              # 123,45 or 123.45
            r'\d+,\d{2}',                  # 123,45
            r'\d+\.\d{2}',                 # 123.45
            r'\d+[,\.]\d{1}',              # 123,4 or 123.4
        ]
        
        # Look for lines that might contain transactions
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue
                
            # Check if line contains both date and amount patterns
            has_date = any(re.search(pattern, line) for pattern in date_patterns)
            has_amount = any(re.search(pattern, line) for pattern in amount_patterns)
            
            if has_date and has_amount:
                patterns.append(f"Line {line_num}: {line}")
            elif has_date or has_amount:
                patterns.append(f"Line {line_num} (partial): {line}")
        
        return patterns
    
    def compare_successful_vs_failed(self, successful_statements: List[int], failed_statements: List[int]):
        """Compare successful vs failed statements to identify differences"""
        print(f"\n{'='*80}")
        print(f"COMPARING SUCCESSFUL VS FAILED STATEMENTS")
        print(f"{'='*80}")
        
        successful_results = []
        failed_results = []
        
        # Analyze successful statements
        print(f"\nANALYZING SUCCESSFUL STATEMENTS:")
        for stmt_id in successful_statements:
            image_path = f"bank statement generator/bank_statements/statement_{stmt_id}.png"
            if os.path.exists(image_path):
                result = self.analyze_statement(image_path, stmt_id)
                successful_results.append(result)
        
        # Analyze failed statements
        print(f"\nANALYZING FAILED STATEMENTS:")
        for stmt_id in failed_statements:
            image_path = f"bank statement generator/bank_statements/statement_{stmt_id}.png"
            if os.path.exists(image_path):
                result = self.analyze_statement(image_path, stmt_id)
                failed_results.append(result)
        
        # Compare patterns
        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        if successful_results:
            avg_text_length_success = sum(r['ocr_text_length'] for r in successful_results) / len(successful_results)
            avg_transactions_success = sum(r['extracted_transactions'] for r in successful_results) / len(successful_results)
            print(f"Successful statements - Avg OCR text length: {avg_text_length_success:.1f}, Avg transactions: {avg_transactions_success:.1f}")
        
        if failed_results:
            avg_text_length_failed = sum(r['ocr_text_length'] for r in failed_results) / len(failed_results)
            avg_potential_patterns_failed = sum(len(r['potential_patterns']) for r in failed_results) / len(failed_results)
            print(f"Failed statements - Avg OCR text length: {avg_text_length_failed:.1f}, Avg potential patterns: {avg_potential_patterns_failed:.1f}")
        
        return successful_results, failed_results

def main():
    diagnostic = ZeroExtractionDiagnostic()
    
    # Based on the previous results, let's analyze some specific cases
    # Successful: statement_2.png (3 transactions), statement_10.png (3 transactions)
    # Failed: statement_1.png (0 transactions)
    
    successful_statements = [2, 10]  # Known to extract transactions
    failed_statements = [1, 3, 4, 5]  # Likely to extract 0 transactions
    
    successful_results, failed_results = diagnostic.compare_successful_vs_failed(
        successful_statements, failed_statements
    )
    
    # Save detailed analysis to file
    with open('zero_extraction_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("ZERO EXTRACTION DIAGNOSTIC ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write("SUCCESSFUL STATEMENTS:\n")
        f.write("-"*30 + "\n")
        for result in successful_results:
            f.write(f"\nStatement {result['statement_id']}:\n")
            f.write(f"  Transactions extracted: {result['extracted_transactions']}\n")
            f.write(f"  OCR text length: {result['ocr_text_length']}\n")
            f.write(f"  OCR text preview: {result['ocr_text'][:200]}...\n")
        
        f.write("\n\nFAILED STATEMENTS:\n")
        f.write("-"*30 + "\n")
        for result in failed_results:
            f.write(f"\nStatement {result['statement_id']}:\n")
            f.write(f"  Transactions extracted: {result['extracted_transactions']}\n")
            f.write(f"  OCR text length: {result['ocr_text_length']}\n")
            f.write(f"  Potential patterns: {len(result['potential_patterns'])}\n")
            f.write(f"  OCR text preview: {result['ocr_text'][:200]}...\n")
            if result['potential_patterns']:
                f.write(f"  Potential patterns found:\n")
                for pattern in result['potential_patterns'][:5]:  # First 5 patterns
                    f.write(f"    - {pattern}\n")
            f.write(f"  Full OCR text:\n{result['ocr_text']}\n")
            f.write("-"*50 + "\n")
    
    print(f"\nDetailed analysis saved to 'zero_extraction_analysis.txt'")

if __name__ == "__main__":
    main()
