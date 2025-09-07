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

class ImprovedBankStatementParser:
    def __init__(self, ground_truth_path: str):
        self.ground_truth_path = ground_truth_path
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Enhanced regex patterns for different transaction formats
        self.transaction_patterns = [
            # Pattern 1: Single line format - DD.MM. DD.MM. DESCRIPTION AMOUNT-
            r'(\d{1,2}\.\d{1,2})\.\s+(\d{1,2}\.\d{1,2})\.\s+(.+?)\s+(\d+(?:[,\.]\d{1,2})?)-?',
            
            # Pattern 2: Compact single line - DD.MM. DD.MM. DESCRIPTION AMOUNT-
            r'(\d{1,2}\.\d{1,2})\.\s*(\d{1,2}\.\d{1,2})\.\s*(.+?)\s+(\d+[,\.]\d{1,2})-',
            
            # Pattern 3: More flexible single line
            r'(\d{1,2}\.\d{1,2})\s+(\d{1,2}\.\d{1,2})\s+(.+?)\s+(\d+[,\.]\d{1,2})-?',
        ]
        
        # OCR configurations to try
        self.ocr_configs = [
            '--oem 3 --psm 6 -l eng',
            '--oem 3 --psm 4 -l eng',
            '--oem 3 --psm 3 -l eng',
            '--oem 1 --psm 6 -l eng',
            '--oem 3 --psm 6 -l deu',
            '--oem 3 --psm 8 -l eng'
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
        
        # Variant 3: Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        variants.append(morph)
        
        # Variant 4: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(enhanced_thresh)
        
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
    
    def parse_multiline_transactions(self, text: str, statement_id: int) -> List[Dict]:
        """Parse transactions that are split across multiple lines"""
        transactions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for date pattern at start of line
            date_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?$', line)
            if date_match and i + 3 < len(lines):
                date1 = date_match.group(1)
                
                # Check if next line is also a date
                next_line = lines[i + 1]
                date2_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?$', next_line)
                
                if date2_match:
                    date2 = date2_match.group(1)
                    
                    # Look for description in next line(s)
                    description_parts = []
                    j = i + 2
                    amount = None
                    
                    # Collect description and find amount
                    while j < len(lines) and j < i + 6:  # Look ahead max 6 lines
                        current_line = lines[j]
                        
                        # Check if this line contains an amount
                        amount_match = re.search(r'(\d+[,\.]\d{1,2})-?$', current_line)
                        if amount_match:
                            amount_str = amount_match.group(1)
                            # Remove amount from description
                            desc_part = re.sub(r'\s*\d+[,\.]\d{1,2}-?$', '', current_line).strip()
                            if desc_part:
                                description_parts.append(desc_part)
                            amount = amount_str
                            break
                        else:
                            # This line is part of description
                            if current_line and not re.match(r'^\d{1,2}\.\d{1,2}\.?$', current_line):
                                description_parts.append(current_line)
                        j += 1
                    
                    if amount and description_parts:
                        description = ' '.join(description_parts)
                        
                        # Parse amount
                        try:
                            amount_value = float(amount.replace(',', '.'))
                            # Make negative (bank statements show debits as positive)
                            amount_value = -amount_value
                        except ValueError:
                            amount_value = 0.0
                        
                        # Parse date (add current year)
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
        """Enhanced transaction parsing with multiple approaches"""
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
                        # Make negative (bank statements show debits as positive)
                        amount_value = -amount_value
                    except ValueError:
                        continue
                    
                    # Parse date (add current year)
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
            transactions = self.parse_multiline_transactions(text, statement_id)
        
        return transactions
    
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
    
    def parse_all_statements(self, statements_dir: str) -> Tuple[List[Dict], str]:
        """Parse all bank statement images in the directory"""
        all_transactions = []
        report_lines = []
        
        # Get all PNG files
        png_files = [f for f in os.listdir(statements_dir) if f.endswith('.png')]
        png_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        report_lines.append("IMPROVED BANK STATEMENT PARSING REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Total statements to process: {len(png_files)}")
        report_lines.append("")
        
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
                        ocr_text = self.extract_text_with_multiple_configs(image_variants)
                        transactions = self.enhanced_parse_transactions_from_text(ocr_text, statement_id)
                        
                        all_transactions.extend(transactions)
                        
                        report_lines.append(f"Statement {statement_id}: {len(transactions)} transactions extracted")
                        if transactions:
                            for trans in transactions:
                                report_lines.append(f"  - {trans['date']}: {trans['amount']:.2f} EUR ({trans['partner']})")
                        
                    except Exception as e:
                        report_lines.append(f"Statement {statement_id}: ERROR - {str(e)}")
                        logging.error(f"Error processing {filename}: {str(e)}")
        
        report_lines.append("")
        report_lines.append(f"Total transactions extracted: {len(all_transactions)}")
        
        return all_transactions, "\n".join(report_lines)
    
    def calculate_accuracy(self, extracted_transactions: List[Dict]) -> Dict:
        """Calculate accuracy metrics by comparing with ground truth"""
        if extracted_transactions:
            extracted_df = pd.DataFrame(extracted_transactions)
        else:
            extracted_df = pd.DataFrame(columns=['statement_id', 'date', 'amount', 'partner', 'category'])
        
        # Merge with ground truth
        merged = pd.merge(
            self.ground_truth_df, 
            extracted_df, 
            on='statement_id', 
            how='left', 
            suffixes=('_true', '_pred')
        )
        
        # Calculate metrics
        total_ground_truth = len(self.ground_truth_df)
        total_extracted = len(extracted_df)
        
        # Date accuracy
        date_matches = merged['date'] == merged['date_pred']
        date_accuracy = date_matches.sum() / total_ground_truth if total_ground_truth > 0 else 0
        
        # Amount accuracy (within 0.01 tolerance)
        amount_matches = abs(merged['amount'] - merged['amount_pred'].fillna(0)) < 0.01
        amount_accuracy = amount_matches.sum() / total_ground_truth if total_ground_truth > 0 else 0
        
        # Partner accuracy
        partner_matches = merged['identified_partner'] == merged['partner_pred']
        partner_accuracy = partner_matches.sum() / total_ground_truth if total_ground_truth > 0 else 0
        
        # Category accuracy
        category_matches = merged['category'] == merged['category_pred']
        category_accuracy = category_matches.sum() / total_ground_truth if total_ground_truth > 0 else 0
        
        # Overall accuracy (all fields correct)
        overall_matches = date_matches & amount_matches & partner_matches & category_matches
        overall_accuracy = overall_matches.sum() / total_ground_truth if total_ground_truth > 0 else 0
        
        # Recall (how many ground truth transactions were found)
        recall = total_extracted / total_ground_truth if total_ground_truth > 0 else 0
        
        return {
            'total_ground_truth': total_ground_truth,
            'total_extracted': total_extracted,
            'overall_accuracy': overall_accuracy,
            'date_accuracy': date_accuracy,
            'amount_accuracy': amount_accuracy,
            'partner_accuracy': partner_accuracy,
            'category_accuracy': category_accuracy,
            'recall': recall
        }

def main():
    # Initialize parser
    parser = ImprovedBankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
    
    # Parse all statements
    statements_dir = "bank statement generator/bank_statements"
    transactions, parsing_report = parser.parse_all_statements(statements_dir)
    
    # Calculate accuracy
    accuracy_metrics = parser.calculate_accuracy(transactions)
    
    # Generate comprehensive report
    report = []
    report.append(parsing_report)
    report.append("\n" + "=" * 50)
    report.append("ACCURACY METRICS")
    report.append("=" * 50)
    report.append(f"Total ground truth transactions: {accuracy_metrics['total_ground_truth']}")
    report.append(f"Total extracted transactions: {accuracy_metrics['total_extracted']}")
    report.append(f"Recall: {accuracy_metrics['recall']:.2%}")
    report.append(f"Overall accuracy: {accuracy_metrics['overall_accuracy']:.2%}")
    report.append(f"Date accuracy: {accuracy_metrics['date_accuracy']:.2%}")
    report.append(f"Amount accuracy: {accuracy_metrics['amount_accuracy']:.2%}")
    report.append(f"Partner accuracy: {accuracy_metrics['partner_accuracy']:.2%}")
    report.append(f"Category accuracy: {accuracy_metrics['category_accuracy']:.2%}")
    
    # Save report
    with open('improved_parsing_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    print(f"\nDetailed report saved to 'improved_parsing_report.txt'")

if __name__ == "__main__":
    main()
