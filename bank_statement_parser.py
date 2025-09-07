import os
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Tesseract path and environment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

class BankStatementParser:
    """
    OCR-based parser for German bank statement images.
    Extracts transaction data and compares with ground truth for accuracy assessment.
    """
    
    def __init__(self, ground_truth_path: str):
        """
        Initialize the parser with ground truth data.
        
        Args:
            ground_truth_path: Path to the ground truth CSV file
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Partner identification patterns
        self.partner_patterns = {
            'ALLIANZ SE': r'ALLIANZ\s+SE',
            'VODAFONE GMBH': r'VODAFONE\s+GMBH',
            'ZEUS BODYPOWER': r'ZEUS\s+BODYPOWER',
            'Stadtwerke Rosenheim': r'Stadtwerke\s+Rosenheim',
            'EDEKA': r'EDEKA',
            'REWE': r'REWE',
            'LIDL': r'LIDL',
            'ALDI SUED': r'ALDI\s+SUED',
            'AMAZON.DE': r'AMAZON\.DE',
            'ZALANDO': r'ZALANDO',
            'EBAY': r'EBAY',
            'SHELL': r'SHELL',
            'ARAL': r'ARAL',
            'JET': r'JET',
            'BURGER KING': r'BURGER\s+KING',
            'MCDONALDS': r'MCDONALDS',
            'PAYPAL': r'PAYPAL'
        }
        
        # Category mapping
        self.category_mapping = {
            'ALLIANZ SE': 'Insurance',
            'VODAFONE GMBH': 'Internet',
            'ZEUS BODYPOWER': 'Gym',
            'Stadtwerke Rosenheim': 'Electricity',
            'EDEKA': 'Groceries',
            'REWE': 'Groceries',
            'LIDL': 'Groceries',
            'ALDI SUED': 'Groceries',
            'AMAZON.DE': 'Shopping',
            'ZALANDO': 'Shopping',
            'EBAY': 'Shopping',
            'SHELL': 'Fuel',
            'ARAL': 'Fuel',
            'JET': 'Fuel',
            'BURGER KING': 'Dining',
            'MCDONALDS': 'Dining',
            'PAYPAL': 'General'
        }
        
        # OCR configuration - using English since German pack not available
        self.ocr_config = '--oem 3 --psm 6 -l eng'
        
        # Results storage
        self.parsed_results = []
        self.accuracy_report = {}
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the bank statement image for better OCR results.
        
        Args:
            image_path: Path to the bank statement image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from bank statement image using OCR.
        
        Args:
            image_path: Path to the bank statement image
            
        Returns:
            Extracted text as string
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image, config=self.ocr_config)
            
            logger.info(f"Successfully extracted text from {image_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse German date format (DD.MM.YYYY, DD.MM., or DD.MM) to standard format.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed date in DD.MM.YYYY format or None if parsing fails
        """
        try:
            # Clean the date string
            date_str = date_str.strip()
            
            # Handle DD.MM. format (add current year)
            if re.match(r'^\d{2}\.\d{2}\.$', date_str):
                current_year = datetime.now().year
                date_str = date_str[:-1] + '.' + str(current_year)
            
            # Handle DD.MM format (add current year)
            elif re.match(r'^\d{2}\.\d{2}$', date_str):
                current_year = datetime.now().year
                date_str = date_str + '.' + str(current_year)
            
            # Validate DD.MM.YYYY format
            if re.match(r'^\d{2}\.\d{2}\.\d{4}$', date_str):
                # Try to parse to validate
                datetime.strptime(date_str, '%d.%m.%Y')
                return date_str
            
            return None
            
        except Exception:
            return None
    
    def parse_amount(self, amount_str: str) -> Optional[float]:
        """
        Parse German amount format (e.g., "123,45" or "123.45") to float.
        
        Args:
            amount_str: Amount string to parse
            
        Returns:
            Parsed amount as negative float (expenses) or None if parsing fails
        """
        try:
            # Clean the amount string
            amount_str = amount_str.strip()
            
            # Handle different formats: "123,45", "123.45", "123,45-"
            if amount_str.endswith('-'):
                amount_str = amount_str[:-1]
            
            # Convert comma to dot for decimal separator
            amount_str = amount_str.replace(',', '.')
            
            # Parse as float and make negative (bank statements show expenses)
            return -float(amount_str)
            
        except Exception:
            return None
    
    def identify_partner(self, description: str) -> str:
        """
        Identify the transaction partner from the description.
        
        Args:
            description: Transaction description
            
        Returns:
            Identified partner name
        """
        description_upper = description.upper()
        
        for partner, pattern in self.partner_patterns.items():
            if re.search(pattern, description_upper):
                return partner
        
        # Fallback: extract first meaningful word
        words = description.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                return word.upper()
        
        return "UNKNOWN"
    
    def classify_category(self, partner: str, description: str) -> str:
        """
        Classify the transaction category based on partner and description.
        
        Args:
            partner: Identified partner
            description: Transaction description
            
        Returns:
            Transaction category
        """
        if partner in self.category_mapping:
            return self.category_mapping[partner]
        
        # Fallback classification based on description keywords
        description_upper = description.upper()
        
        if any(keyword in description_upper for keyword in ['TANKSTELLE', 'SHELL', 'ARAL', 'JET']):
            return 'Fuel'
        elif any(keyword in description_upper for keyword in ['BURGER', 'MCDONALDS', 'RESTAURANT']):
            return 'Dining'
        elif any(keyword in description_upper for keyword in ['EDEKA', 'REWE', 'LIDL', 'ALDI']):
            return 'Groceries'
        elif any(keyword in description_upper for keyword in ['AMAZON', 'ZALANDO', 'EBAY']):
            return 'Shopping'
        
        return 'General'
    
    def extract_statement_id(self, text: str, image_path: str) -> int:
        """
        Extract statement ID from the image path or text.
        
        Args:
            text: Extracted text from image
            image_path: Path to the image file
            
        Returns:
            Statement ID
        """
        # Extract from filename (e.g., statement_1.png -> 1)
        filename = os.path.basename(image_path)
        match = re.search(r'statement_(\d+)', filename)
        if match:
            return int(match.group(1))
        
        # Fallback: try to extract from text
        match = re.search(r'AUSZUG\s+(\d+)', text)
        if match:
            return int(match.group(1))
        
        return 0
    
    def parse_transactions_from_text(self, text: str, statement_id: int) -> List[Dict]:
        """
        Parse transaction data from extracted text.
        
        Args:
            text: Extracted text from OCR
            statement_id: Statement ID
            
        Returns:
            List of parsed transactions
        """
        transactions = []
        lines = text.split('\n')
        
        # Look for transaction patterns based on actual OCR output
        # Format: DD.MM, DD.MM, DESCRIPTION AMOUNT. or AMOUNT,
        # Handle both 2-decimal and whole number amounts
        transaction_pattern = r'(\d{2}\.\d{2}),\s+(\d{2}\.\d{2}),\s+(.+?)\s+(\d+(?:[,\.]\d{1,2})?)[,.]?'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match transaction pattern
            match = re.search(transaction_pattern, line)
            if match:
                beleg_date = match.group(1)
                valuta_date = match.group(2)
                description = match.group(3).strip()
                amount_str = match.group(4)
                
                # Parse components
                parsed_date = self.parse_date(beleg_date)
                parsed_amount = self.parse_amount(amount_str)
                
                if parsed_date and parsed_amount is not None:
                    partner = self.identify_partner(description)
                    category = self.classify_category(partner, description)
                    
                    transaction = {
                        'statement_id': statement_id,
                        'date': parsed_date,
                        'description': description,
                        'amount': parsed_amount,
                        'identified_partner': partner,
                        'category': category
                    }
                    
                    transactions.append(transaction)
        
        return transactions
    
    def parse_single_statement(self, image_path: str) -> List[Dict]:
        """
        Parse a single bank statement image.
        
        Args:
            image_path: Path to the bank statement image
            
        Returns:
            List of parsed transactions
        """
        logger.info(f"Parsing statement: {image_path}")
        
        try:
            # Extract text from image
            text = self.extract_text_from_image(image_path)
            
            if not text:
                logger.warning(f"No text extracted from {image_path}")
                return []
            
            # Extract statement ID
            statement_id = self.extract_statement_id(text, image_path)
            
            # Parse transactions
            transactions = self.parse_transactions_from_text(text, statement_id)
            
            logger.info(f"Extracted {len(transactions)} transactions from statement {statement_id}")
            return transactions
            
        except Exception as e:
            logger.error(f"Error parsing {image_path}: {str(e)}")
            return []
    
    def parse_all_statements(self, images_folder: str) -> pd.DataFrame:
        """
        Parse all bank statement images in a folder.
        
        Args:
            images_folder: Path to folder containing bank statement images
            
        Returns:
            DataFrame with all parsed transactions
        """
        all_transactions = []
        
        # Get all PNG files in the folder
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
        image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        logger.info(f"Found {len(image_files)} bank statement images to process")
        
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transactions = self.parse_single_statement(image_path)
            all_transactions.extend(transactions)
        
        # Convert to DataFrame
        if all_transactions:
            self.parsed_results = pd.DataFrame(all_transactions)
            logger.info(f"Successfully parsed {len(all_transactions)} total transactions")
        else:
            self.parsed_results = pd.DataFrame(columns=['statement_id', 'date', 'description', 'amount', 'identified_partner', 'category'])
            logger.warning("No transactions were successfully parsed")
        
        return self.parsed_results
    
    def calculate_accuracy(self) -> Dict:
        """
        Calculate accuracy metrics by comparing parsed results with ground truth.
        
        Returns:
            Dictionary containing accuracy metrics
        """
        if self.parsed_results.empty:
            return {"error": "No parsed results to compare"}
        
        # Merge parsed results with ground truth on statement_id and date
        merged = pd.merge(
            self.ground_truth_df, 
            self.parsed_results, 
            on=['statement_id'], 
            suffixes=('_truth', '_parsed'),
            how='outer'
        )
        
        # Calculate field-level accuracy
        total_ground_truth = len(self.ground_truth_df)
        total_parsed = len(self.parsed_results)
        
        # Find matching transactions (same statement_id and similar date/amount)
        matches = 0
        date_matches = 0
        amount_matches = 0
        partner_matches = 0
        category_matches = 0
        description_similarity_scores = []
        
        for _, row in merged.iterrows():
            if pd.notna(row['date_truth']) and pd.notna(row['date_parsed']):
                # Date accuracy
                if row['date_truth'] == row['date_parsed']:
                    date_matches += 1
                
                # Amount accuracy (with small tolerance)
                if pd.notna(row['amount_truth']) and pd.notna(row['amount_parsed']):
                    if abs(row['amount_truth'] - row['amount_parsed']) < 0.01:
                        amount_matches += 1
                
                # Partner accuracy
                if pd.notna(row['identified_partner_truth']) and pd.notna(row['identified_partner_parsed']):
                    if row['identified_partner_truth'] == row['identified_partner_parsed']:
                        partner_matches += 1
                
                # Category accuracy
                if pd.notna(row['category_truth']) and pd.notna(row['category_parsed']):
                    if row['category_truth'] == row['category_parsed']:
                        category_matches += 1
                
                # Description similarity
                if pd.notna(row['description_truth']) and pd.notna(row['description_parsed']):
                    similarity = SequenceMatcher(None, row['description_truth'], row['description_parsed']).ratio()
                    description_similarity_scores.append(similarity)
                    
                    # Count as match if similarity > 0.8
                    if similarity > 0.8:
                        matches += 1
        
        # Calculate accuracy percentages
        accuracy_report = {
            'total_ground_truth_transactions': total_ground_truth,
            'total_parsed_transactions': total_parsed,
            'parsing_recall': (total_parsed / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'overall_accuracy': (matches / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'date_accuracy': (date_matches / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'amount_accuracy': (amount_matches / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'partner_accuracy': (partner_matches / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'category_accuracy': (category_matches / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'description_similarity_avg': np.mean(description_similarity_scores) if description_similarity_scores else 0,
            'successful_matches': matches,
            'failed_extractions': total_ground_truth - total_parsed
        }
        
        self.accuracy_report = accuracy_report
        return accuracy_report
    
    def generate_report(self, output_path: str = "parsing_accuracy_report.txt"):
        """
        Generate a detailed accuracy report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.accuracy_report:
            logger.error("No accuracy report available. Run calculate_accuracy() first.")
            return
        
        report_content = f"""
BANK STATEMENT PARSER - ACCURACY REPORT
=====================================

OVERALL PERFORMANCE:
- Total Ground Truth Transactions: {self.accuracy_report['total_ground_truth_transactions']}
- Total Parsed Transactions: {self.accuracy_report['total_parsed_transactions']}
- Parsing Recall: {self.accuracy_report['parsing_recall']:.2f}%
- Overall Accuracy: {self.accuracy_report['overall_accuracy']:.2f}%

FIELD-LEVEL ACCURACY:
- Date Accuracy: {self.accuracy_report['date_accuracy']:.2f}%
- Amount Accuracy: {self.accuracy_report['amount_accuracy']:.2f}%
- Partner Identification Accuracy: {self.accuracy_report['partner_accuracy']:.2f}%
- Category Classification Accuracy: {self.accuracy_report['category_accuracy']:.2f}%
- Description Similarity (Average): {self.accuracy_report['description_similarity_avg']:.2f}

EXTRACTION RESULTS:
- Successful Matches: {self.accuracy_report['successful_matches']}
- Failed Extractions: {self.accuracy_report['failed_extractions']}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Accuracy report saved to: {output_path}")
        print(report_content)
    
    def save_parsed_results(self, output_path: str = "parsed_results.csv"):
        """
        Save parsed results to CSV file.
        
        Args:
            output_path: Path to save the parsed results
        """
        if not self.parsed_results.empty:
            self.parsed_results.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Parsed results saved to: {output_path}")
        else:
            logger.warning("No parsed results to save")


def main():
    """
    Main function to run the bank statement parser.
    """
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
    
    print("🚀 Starting Bank Statement Parser...")
    
    # Initialize parser
    parser = BankStatementParser(GROUND_TRUTH_PATH)
    
    # Parse all statements
    print("📄 Parsing bank statement images...")
    parsed_df = parser.parse_all_statements(IMAGES_FOLDER)
    
    # Save parsed results
    parser.save_parsed_results("parsed_bank_statements.csv")
    
    # Calculate accuracy
    print("📊 Calculating accuracy metrics...")
    accuracy = parser.calculate_accuracy()
    
    # Generate report
    parser.generate_report("bank_statement_parsing_report.txt")
    
    print("✅ Bank statement parsing completed!")
    print(f"📈 Overall Accuracy: {accuracy.get('overall_accuracy', 0):.2f}%")
    print(f"📋 Parsed {accuracy.get('total_parsed_transactions', 0)} out of {accuracy.get('total_ground_truth_transactions', 0)} transactions")


if __name__ == "__main__":
    main()
