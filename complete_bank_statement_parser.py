import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class CompleteBankStatementParser:
    def __init__(self, ground_truth_path: str):
        self.ground_truth_path = ground_truth_path
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        
        # More precise regex patterns to avoid duplicates
        self.transaction_patterns = [
            # Pattern 1: Single line format - DD.MM. DD.MM. DESCRIPTION AMOUNT-
            r'(\d{1,2}\.\d{1,2})\.\s+(\d{1,2}\.\d{1,2})\.\s+(.+?)\s+(\d+[,\.]\d{1,2})-',
            
            # Pattern 2: Compact single line - DD.MM. DD.MM. DESCRIPTION AMOUNT-
            r'(\d{1,2}\.\d{1,2})\.\s*(\d{1,2}\.\d{1,2})\.\s*(.+?)\s+(\d+[,\.]\d{1,2})-',
        ]
        
        # OCR configurations to try
        self.ocr_configs = [
            '--oem 3 --psm 6 -l eng',
            '--oem 3 --psm 4 -l eng',
            '--oem 3 --psm 3 -l eng',
        ]
        
    def advanced_preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """Create multiple preprocessed variants of the image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Scale to higher resolution for better OCR
        scale_factor = 2
        height, width = gray.shape
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
        
        variants = []
        
        # Variant 1: Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive_thresh)
        
        # Variant 2: OTSU threshold
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu_thresh)
        
        return variants
    
    def extract_text_with_multiple_configs(self, image_variants: List[np.ndarray]) -> str:
        """Try multiple OCR configurations and return the best result"""
        best_text = ""
        best_confidence = 0
        
        for variant in image_variants:
            for config in self.ocr_configs:
                try:
                    # Get OCR result with confidence
                    data = pytesseract.image_to_data(variant, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_text = pytesseract.image_to_string(variant, config=config)
                except Exception as e:
                    continue
        
        return best_text, best_confidence
    
    def parse_multiline_transactions_precise(self, text: str, statement_id: int) -> List[Dict]:
        """Parse transactions that are split across multiple lines with precise matching"""
        transactions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for transaction blocks between KONTOSTAND markers
        in_transaction_section = False
        transaction_lines = []
        
        for line in lines:
            if 'KONTOSTAND' in line.upper() and any(char.isdigit() for char in line):
                if in_transaction_section:
                    # Process accumulated transaction lines
                    transactions.extend(self.extract_transactions_from_block(transaction_lines, statement_id))
                    transaction_lines = []
                in_transaction_section = True
            elif 'ABRECHNUNGSTERMIN' in line.upper():
                if in_transaction_section:
                    # Process final transaction block
                    transactions.extend(self.extract_transactions_from_block(transaction_lines, statement_id))
                in_transaction_section = False
                break
            elif in_transaction_section:
                transaction_lines.append(line)
        
        return transactions
    
    def extract_transactions_from_block(self, lines: List[str], statement_id: int) -> List[Dict]:
        """Extract transactions from a block of lines"""
        transactions = []
        i = 0
        
        while i < len(lines):
            # Look for date pattern
            date_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?\s*$', lines[i])
            if date_match and i + 1 < len(lines):
                date1 = date_match.group(1)
                
                # Check next line for second date
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                date2_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?\s*$', next_line)
                
                if date2_match and i + 2 < len(lines):
                    date2 = date2_match.group(1)
                    
                    # Look for description and amount in subsequent lines
                    description_parts = []
                    amount = None
                    j = i + 2
                    
                    while j < len(lines) and j < i + 5:  # Look ahead max 3 more lines
                        current_line = lines[j]
                        
                        # Check if this line contains an amount at the end
                        amount_match = re.search(r'(\d+[,\.]\d{1,2})-?\s*$', current_line)
                        if amount_match:
                            amount_str = amount_match.group(1)
                            # Remove amount from description
                            desc_part = re.sub(r'\s*\d+[,\.]\d{1,2}-?\s*$', '', current_line).strip()
                            if desc_part:
                                description_parts.append(desc_part)
                            amount = amount_str
                            break
                        else:
                            # This line is part of description
                            if current_line and not re.match(r'^\d{1,2}\.\d{1,2}\.?\s*$', current_line):
                                description_parts.append(current_line)
                        j += 1
                    
                    if amount and description_parts:
                        description = ' '.join(description_parts)
                        
                        # Parse amount
                        try:
                            amount_value = float(amount.replace(',', '.'))
                            amount_value = -amount_value  # Make negative
                        except ValueError:
                            amount_value = 0.0
                        
                        # Parse date
                        try:
                            current_year = datetime.now().year
                            date_obj = datetime.strptime(f"{date1}.{current_year}", "%d.%m.%Y")
                            formatted_date = date_obj.strftime("%d.%m.%Y")
                        except ValueError:
                            formatted_date = f"{date1}.{datetime.now().year}"
                        
                        transaction = {
                            'statement_id': statement_id,
                            'date': formatted_date,
                            'amount': amount_value,
                            'partner': self.identify_partner(description),
                            'category': self.classify_category(description),
                            'description': description
                        }
                        transactions.append(transaction)
                        
                        # Skip processed lines
                        i = j
                        continue
            
            i += 1
        
        return transactions
    
    def enhanced_parse_transactions_from_text(self, text: str, statement_id: int) -> List[Dict]:
        """Enhanced transaction parsing with precise duplicate prevention"""
        transactions = []
        
        # First try single-line patterns
        for pattern in self.transaction_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                if len(match) >= 4:
                    date1, date2, description, amount_str = match[:4]
                    
                    # Parse amount
                    try:
                        amount_value = float(amount_str.replace(',', '.'))
                        amount_value = -amount_value
                    except ValueError:
                        continue
                    
                    # Parse date
                    try:
                        current_year = datetime.now().year
                        date_obj = datetime.strptime(f"{date1}.{current_year}", "%d.%m.%Y")
                        formatted_date = date_obj.strftime("%d.%m.%Y")
                    except ValueError:
                        formatted_date = f"{date1}.{datetime.now().year}"
                    
                    transaction = {
                        'statement_id': statement_id,
                        'date': formatted_date,
                        'amount': amount_value,
                        'partner': self.identify_partner(description),
                        'category': self.classify_category(description),
                        'description': description.strip()
                    }
                    transactions.append(transaction)
        
        # If no single-line transactions found, try multiline parsing
        if not transactions:
            transactions = self.parse_multiline_transactions_precise(text, statement_id)
        
        # Remove duplicates based on date and amount
        unique_transactions = []
        seen = set()
        
        for trans in transactions:
            key = (trans['date'], trans['amount'])
            if key not in seen:
                seen.add(key)
                unique_transactions.append(trans)
        
        return unique_transactions
    
    def identify_partner(self, description: str) -> str:
        """Identify transaction partner from description"""
        description = description.upper()
        
        partner_patterns = {
            'ZALANDO': r'ZALANDO',
            'AMAZON': r'AMAZON',
            'PAYPAL': r'PAYPAL',
            'VODAFONE': r'VODAFONE',
            'BURGER KING': r'BURGER\s*KING',
            'ALLIANZ': r'ALLIANZ',
            'ZEUS BODYPOWER': r'ZEUS\s*BODYPOWER',
            'SHELL': r'SHELL',
            'JET': r'JET',
            'ALDI': r'ALDI',
            'STADTWERKE': r'STADTWERKE'
        }
        
        for partner, pattern in partner_patterns.items():
            if re.search(pattern, description):
                return partner
        
        return "Unknown"
    
    def classify_category(self, description: str) -> str:
        """Classify transaction category"""
        description = description.upper()
        
        if any(word in description for word in ['ZALANDO', 'AMAZON', 'SHOPPING']):
            return 'Shopping'
        elif any(word in description for word in ['BURGER', 'RESTAURANT', 'FOOD']):
            return 'Food'
        elif any(word in description for word in ['VODAFONE', 'TELEFON', 'MOBILE']):
            return 'Telecommunications'
        elif any(word in description for word in ['BODYPOWER', 'GYM', 'FITNESS']):
            return 'Fitness'
        elif any(word in description for word in ['SHELL', 'JET', 'TANKSTELLE', 'FUEL']):
            return 'Fuel'
        elif any(word in description for word in ['ALLIANZ', 'INSURANCE', 'VERSICHERUNG']):
            return 'Insurance'
        elif any(word in description for word in ['STROM', 'STADTWERKE', 'UTILITIES']):
            return 'Utilities'
        else:
            return 'Other'

    def process_all_statements(self, statements_dir: str) -> Dict:
        """Process all bank statement images and return comprehensive results"""
        
        # Get all PNG files
        png_files = [f for f in os.listdir(statements_dir) if f.endswith('.png')]
        png_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        print(f"PROCESSING ALL {len(png_files)} BANK STATEMENTS")
        print("=" * 60)
        
        all_results = {}
        all_transactions = []
        total_confidence = 0
        processed_count = 0
        
        perfect_matches = 0
        partial_matches = 0
        zero_extractions = 0
        
        print(f"{'Statement':<10} {'GT':<4} {'Ext':<4} {'Conf%':<6} {'Status'}")
        print("-" * 40)
        
        for filename in png_files:
            if filename.endswith('.png'):
                # Extract statement ID
                match = re.search(r'statement_(\d+)\.png', filename)
                if match:
                    statement_id = int(match.group(1))
                    image_path = os.path.join(statements_dir, filename)
                    
                    try:
                        # Process the image
                        image_variants = self.advanced_preprocess_image(image_path)
                        ocr_text, confidence = self.extract_text_with_multiple_configs(image_variants)
                        transactions = self.enhanced_parse_transactions_from_text(ocr_text, statement_id)
                        
                        # Get ground truth count
                        gt_count = len(self.ground_truth_df[self.ground_truth_df['statement_id'] == statement_id])
                        extracted_count = len(transactions)
                        
                        # Determine status
                        if extracted_count == gt_count:
                            status = "✓ Perfect"
                            perfect_matches += 1
                        elif extracted_count > 0:
                            status = "~ Partial"
                            partial_matches += 1
                        else:
                            status = "✗ Zero"
                            zero_extractions += 1
                        
                        print(f"{statement_id:<10} {gt_count:<4} {extracted_count:<4} {confidence:<6.1f} {status}")
                        
                        all_results[statement_id] = {
                            'ground_truth': gt_count,
                            'extracted': extracted_count,
                            'confidence': confidence,
                            'status': status
                        }
                        
                        all_transactions.extend(transactions)
                        total_confidence += confidence
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"{statement_id:<10} ERROR: {str(e)[:20]}...")
                        all_results[statement_id] = {
                            'ground_truth': len(self.ground_truth_df[self.ground_truth_df['statement_id'] == statement_id]),
                            'extracted': 0,
                            'confidence': 0,
                            'status': 'ERROR'
                        }
        
        # Calculate summary statistics
        total_gt = sum(r['ground_truth'] for r in all_results.values())
        total_extracted = sum(r['extracted'] for r in all_results.values())
        avg_confidence = total_confidence / processed_count if processed_count > 0 else 0
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total statements processed: {len(all_results)}")
        print(f"Ground truth transactions: {total_gt}")
        print(f"Extracted transactions: {total_extracted}")
        print(f"Overall extraction rate: {total_extracted/total_gt:.1%}")
        print(f"Average OCR confidence: {avg_confidence:.1f}%")
        print()
        print(f"Perfect matches (100%): {perfect_matches} statements ({perfect_matches/len(all_results):.1%})")
        print(f"Partial matches (>0%):  {partial_matches} statements ({partial_matches/len(all_results):.1%})")
        print(f"Zero extractions (0%):  {zero_extractions} statements ({zero_extractions/len(all_results):.1%})")
        print(f"Statement-level success: {(perfect_matches + partial_matches)/len(all_results):.1%}")
        
        return {
            'results': all_results,
            'transactions': all_transactions,
            'summary': {
                'total_statements': len(all_results),
                'total_gt': total_gt,
                'total_extracted': total_extracted,
                'extraction_rate': total_extracted/total_gt if total_gt > 0 else 0,
                'avg_confidence': avg_confidence,
                'perfect_matches': perfect_matches,
                'partial_matches': partial_matches,
                'zero_extractions': zero_extractions
            }
        }

def main():
    """Process all 200 bank statements"""
    parser = CompleteBankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
    
    # Process all statements
    statements_dir = "bank statement generator/bank_statements"
    results = parser.process_all_statements(statements_dir)
    
    # Save results to CSV
    if results['transactions']:
        df = pd.DataFrame(results['transactions'])
        df.to_csv('complete_parsed_transactions.csv', index=False)
        print(f"\nSaved {len(results['transactions'])} transactions to 'complete_parsed_transactions.csv'")
    
    # Save detailed report
    with open('complete_parsing_report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPLETE BANK STATEMENT PARSING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        summary = results['summary']
        f.write(f"Total statements: {summary['total_statements']}\n")
        f.write(f"Ground truth transactions: {summary['total_gt']}\n")
        f.write(f"Extracted transactions: {summary['total_extracted']}\n")
        f.write(f"Extraction rate: {summary['extraction_rate']:.1%}\n")
        f.write(f"Average OCR confidence: {summary['avg_confidence']:.1f}%\n")
        f.write(f"Perfect matches: {summary['perfect_matches']}\n")
        f.write(f"Partial matches: {summary['partial_matches']}\n")
        f.write(f"Zero extractions: {summary['zero_extractions']}\n")
        
        f.write(f"\nPER-STATEMENT RESULTS:\n")
        f.write("-" * 30 + "\n")
        for stmt_id, result in sorted(results['results'].items()):
            f.write(f"Statement {stmt_id}: {result['extracted']}/{result['ground_truth']} "
                   f"({result['confidence']:.1f}% conf) - {result['status']}\n")
    
    print(f"Detailed report saved to 'complete_parsing_report.txt'")

if __name__ == "__main__":
    main()
