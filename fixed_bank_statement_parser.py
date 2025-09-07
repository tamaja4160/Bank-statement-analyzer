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

class FixedBankStatementParser:
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
        
        logging.info(f"Best OCR confidence: {best_confidence:.2f}")
        return best_text
    
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

def test_fixed_parser():
    """Test the fixed parser on first 5 statements"""
    parser = FixedBankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
    ground_truth = pd.read_csv("bank statement generator/bank_statements/ground_truth.csv")
    
    print("FIXED PARSER TEST RESULTS")
    print("=" * 50)
    print(f"{'Statement':<10} {'Ground Truth':<12} {'Extracted':<10} {'Status'}")
    print("-" * 45)
    
    test_statements = [1, 2, 3, 4, 5]
    results = {}
    
    for stmt_id in test_statements:
        image_path = f"bank statement generator/bank_statements/statement_{stmt_id}.png"
        if os.path.exists(image_path):
            try:
                # Process the image
                image_variants = parser.advanced_preprocess_image(image_path)
                ocr_text = parser.extract_text_with_multiple_configs(image_variants)
                transactions = parser.enhanced_parse_transactions_from_text(ocr_text, stmt_id)
                
                gt_count = len(ground_truth[ground_truth['statement_id'] == stmt_id])
                extracted_count = len(transactions)
                
                status = "✓ Perfect" if extracted_count == gt_count else "~ Partial" if extracted_count > 0 else "✗ Zero"
                
                print(f"{stmt_id:<10} {gt_count:<12} {extracted_count:<10} {status}")
                results[stmt_id] = {'gt': gt_count, 'extracted': extracted_count}
                
            except Exception as e:
                print(f"{stmt_id:<10} Error: {str(e)[:30]}...")
    
    # Summary
    if results:
        total_gt = sum(r['gt'] for r in results.values())
        total_extracted = sum(r['extracted'] for r in results.values())
        print(f"\nFixed Parser Results:")
        print(f"- Ground truth transactions: {total_gt}")
        print(f"- Extracted transactions: {total_extracted}")
        print(f"- Extraction rate: {total_extracted/total_gt:.1%}")

if __name__ == "__main__":
    test_fixed_parser()
