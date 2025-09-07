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

class EnhancedBankStatementParser:
    """
    Enhanced OCR-based parser for German bank statement images with improved preprocessing and configuration.
    """
    
    def __init__(self, ground_truth_path: str):
        """
        Initialize the enhanced parser with ground truth data.
        
        Args:
            ground_truth_path: Path to the ground truth CSV file
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Partner identification patterns (expanded)
        self.partner_patterns = {
            'ALLIANZ SE': r'ALLIANZ\s*SE|ALLIANZ\s*SEK',
            'VODAFONE GMBH': r'VODAFONE\s*GMBH|VODAFONE',
            'ZEUS BODYPOWER': r'ZEUS\s*BODYPOWER|ZEUS',
            'Stadtwerke Rosenheim': r'Stadtwerke\s*Rosenheim|STADTWERKE',
            'EDEKA': r'EDEKA',
            'REWE': r'REWE',
            'LIDL': r'LIDL',
            'ALDI SUED': r'ALDI\s*SUED|ALDI',
            'AMAZON.DE': r'AMAZON\.?DE|AMAZON\s*MKTPLC|AMAZON',
            'ZALANDO': r'ZALANDO',
            'EBAY': r'EBAY',
            'SHELL': r'SHELL',
            'ARAL': r'ARAL',
            'JET': r'JET\s*TANKSTELLE|JET',
            'BURGER KING': r'BURGER\s*KING',
            'MCDONALDS': r'MCDONALDS|MCDONALD',
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
        
        # Multiple OCR configurations to try
        self.ocr_configs = [
            # Try German first if available, fallback to English
            '--oem 1 --psm 6 -l deu -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüßÄÖÜ.,- €() -c preserve_interword_spaces=1',
            '--oem 1 --psm 4 -l deu -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüßÄÖÜ.,- €()',
            '--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüßÄÖÜ.,- €()',
            '--oem 1 --psm 11 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüßÄÖÜ.,- €()',
            '--oem 3 --psm 4 -l eng',
            '--oem 1 --psm 8 -l eng'
        ]
        
        # Results storage
        self.parsed_results = []
        self.accuracy_report = {}
    
    def enhance_image_resolution(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """
        Enhance image resolution by scaling to target DPI.
        
        Args:
            image: Input image
            target_dpi: Target DPI for OCR (300+ recommended)
            
        Returns:
            Scaled image
        """
        # Assume original is 72 DPI, scale to target DPI
        scale_factor = target_dpi / 72.0
        if scale_factor > 1:
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Skew-corrected image
        """
        # Find edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for line in lines[:10]:  # Use first 10 lines
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Only correct if significant skew
                    # Rotate image
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def advanced_preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """
        Apply multiple advanced preprocessing techniques and return multiple versions.
        
        Args:
            image_path: Path to the bank statement image
            
        Returns:
            List of preprocessed image variants
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Enhance resolution first
        image = self.enhance_image_resolution(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Correct skew
        gray = self.correct_skew(gray)
        
        processed_variants = []
        
        # Variant 1: Advanced denoising + CLAHE + Adaptive threshold
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_variants.append(adaptive_thresh)
        
        # Variant 2: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        processed_variants.append(morph)
        
        # Variant 3: OTSU thresholding with blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_variants.append(otsu_thresh)
        
        # Variant 4: Edge enhancement
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_variants.append(sharp_thresh)
        
        return processed_variants
    
    def extract_text_with_multiple_configs(self, image_variants: List[np.ndarray]) -> str:
        """
        Extract text using multiple OCR configurations and image variants.
        
        Args:
            image_variants: List of preprocessed image variants
            
        Returns:
            Best extracted text
        """
        best_text = ""
        best_confidence = 0
        
        for variant_idx, image_variant in enumerate(image_variants):
            for config_idx, config in enumerate(self.ocr_configs):
                try:
                    # Try to get confidence data
                    data = pytesseract.image_to_data(image_variant, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence:
                        text = pytesseract.image_to_string(image_variant, config=config)
                        if text.strip():  # Only update if we got actual text
                            best_text = text
                            best_confidence = avg_confidence
                            logger.debug(f"Best config so far: variant {variant_idx}, config {config_idx}, confidence: {avg_confidence:.2f}")
                
                except Exception as e:
                    # If German language pack not available, skip German configs
                    if 'deu' in config:
                        logger.debug(f"German language pack not available, skipping config: {config}")
                        continue
                    logger.warning(f"OCR failed with config {config}: {str(e)}")
                    continue
        
        logger.info(f"Best OCR confidence: {best_confidence:.2f}")
        return best_text
    
    def enhanced_parse_date(self, date_str: str) -> Optional[str]:
        """
        Enhanced date parsing with multiple format support.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed date in DD.MM.YYYY format or None if parsing fails
        """
        try:
            # Clean the date string
            date_str = re.sub(r'[^\d.]', '', date_str.strip())
            
            # Handle various formats
            patterns = [
                r'^(\d{2})\.(\d{2})\.(\d{4})$',  # DD.MM.YYYY
                r'^(\d{2})\.(\d{2})\.$',        # DD.MM.
                r'^(\d{2})\.(\d{2})$',          # DD.MM
                r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$',  # D.M.YYYY or DD.M.YYYY etc.
                r'^(\d{1,2})\.(\d{1,2})\.$',    # D.M. or DD.M.
                r'^(\d{1,2})\.(\d{1,2})$'       # D.M or DD.M
            ]
            
            for pattern in patterns:
                match = re.match(pattern, date_str)
                if match:
                    day, month = match.group(1), match.group(2)
                    year = match.group(3) if len(match.groups()) == 3 else str(datetime.now().year)
                    
                    # Pad with zeros if needed
                    day = day.zfill(2)
                    month = month.zfill(2)
                    
                    # Validate date
                    try:
                        datetime.strptime(f"{day}.{month}.{year}", '%d.%m.%Y')
                        return f"{day}.{month}.{year}"
                    except ValueError:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def enhanced_parse_amount(self, amount_str: str) -> Optional[float]:
        """
        Enhanced amount parsing with multiple format support.
        
        Args:
            amount_str: Amount string to parse
            
        Returns:
            Parsed amount as negative float (expenses) or None if parsing fails
        """
        try:
            # Clean the amount string
            amount_str = re.sub(r'[^\d,.-]', '', amount_str.strip())
            
            # Handle various German number formats
            patterns = [
                r'^(\d+)[,.](\d{2})[-]?$',      # 123,45 or 123.45 with optional -
                r'^(\d+)[-]?$',                 # 123 (whole numbers)
                r'^(\d{1,3}(?:[.,]\d{3})*)[,.](\d{2})[-]?$',  # 1.234,56 or 1,234.56
            ]
            
            for pattern in patterns:
                match = re.match(pattern, amount_str)
                if match:
                    if len(match.groups()) == 2:
                        # Has decimal part
                        whole, decimal = match.group(1), match.group(2)
                        # Remove thousand separators
                        whole = re.sub(r'[,.]', '', whole)
                        amount = float(f"{whole}.{decimal}")
                    else:
                        # Whole number only
                        whole = re.sub(r'[,.]', '', match.group(1))
                        amount = float(whole)
                    
                    # Make negative (bank statements show expenses)
                    return -abs(amount)
            
            return None
            
        except Exception:
            return None
    
    def enhanced_parse_transactions_from_text(self, text: str, statement_id: int) -> List[Dict]:
        """
        Enhanced transaction parsing with multiple regex patterns.
        
        Args:
            text: Extracted text from OCR
            statement_id: Statement ID
            
        Returns:
            List of parsed transactions
        """
        transactions = []
        lines = text.split('\n')
        
        # Multiple transaction patterns to try
        transaction_patterns = [
            # Original pattern
            r'(\d{1,2}\.\d{1,2}\.?),?\s+(\d{1,2}\.\d{1,2}\.?),?\s+(.+?)\s+(\d+(?:[,\.]\d{1,2})?)[,.]?',
            # More flexible pattern
            r'(\d{1,2}\.\d{1,2}\.?)\s*,?\s*(\d{1,2}\.\d{1,2}\.?)\s*,?\s+(.+?)\s+(\d+[,\.]\d{0,2})[,.-]*',
            # Pattern for lines with more spacing
            r'(\d{1,2}\.\d{1,2}\.?)\s+(\d{1,2}\.\d{1,2}\.?)\s+(.+?)\s+(\d+[,\.]\d{2})',
            # Pattern for amounts at the end
            r'(\d{1,2}\.\d{1,2}\.?)[,\s]+(\d{1,2}\.\d{1,2}\.?)[,\s]+(.+?)\s+(\d+[,\.]\d{1,2})[-,.\s]*$'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:  # Skip very short lines
                continue
            
            # Try each pattern
            for pattern in transaction_patterns:
                match = re.search(pattern, line)
                if match:
                    beleg_date = match.group(1)
                    valuta_date = match.group(2)
                    description = match.group(3).strip()
                    amount_str = match.group(4)
                    
                    # Parse components with enhanced methods
                    parsed_date = self.enhanced_parse_date(beleg_date)
                    parsed_amount = self.enhanced_parse_amount(amount_str)
                    
                    if parsed_date and parsed_amount is not None and description:
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
                        break  # Found a match, don't try other patterns for this line
        
        return transactions
    
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
    
    def parse_single_statement(self, image_path: str) -> List[Dict]:
        """
        Parse a single bank statement image with enhanced processing.
        
        Args:
            image_path: Path to the bank statement image
            
        Returns:
            List of parsed transactions
        """
        logger.info(f"Enhanced parsing statement: {image_path}")
        
        try:
            # Apply advanced preprocessing to get multiple image variants
            image_variants = self.advanced_preprocess_image(image_path)
            
            # Extract text using multiple configurations
            text = self.extract_text_with_multiple_configs(image_variants)
            
            if not text:
                logger.warning(f"No text extracted from {image_path}")
                return []
            
            # Extract statement ID
            statement_id = self.extract_statement_id(text, image_path)
            
            # Parse transactions with enhanced method
            transactions = self.enhanced_parse_transactions_from_text(text, statement_id)
            
            logger.info(f"Enhanced extraction: {len(transactions)} transactions from statement {statement_id}")
            return transactions
            
        except Exception as e:
            logger.error(f"Error in enhanced parsing {image_path}: {str(e)}")
            return []
    
    def parse_all_statements(self, images_folder: str) -> pd.DataFrame:
        """
        Parse all bank statement images in a folder with enhanced processing.
        
        Args:
            images_folder: Path to folder containing bank statement images
            
        Returns:
            DataFrame with all parsed transactions
        """
        all_transactions = []
        
        # Get all PNG files in the folder
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
        image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        logger.info(f"Enhanced processing: Found {len(image_files)} bank statement images")
        
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transactions = self.parse_single_statement(image_path)
            all_transactions.extend(transactions)
        
        # Convert to DataFrame
        if all_transactions:
            self.parsed_results = pd.DataFrame(all_transactions)
            logger.info(f"Enhanced processing: Successfully parsed {len(all_transactions)} total transactions")
        else:
            self.parsed_results = pd.DataFrame(columns=['statement_id', 'date', 'description', 'amount', 'identified_partner', 'category'])
            logger.warning("Enhanced processing: No transactions were successfully parsed")
        
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
    
    def generate_report(self, output_path: str = "enhanced_parsing_accuracy_report.txt"):
        """
        Generate a detailed accuracy report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.accuracy_report:
            logger.error("No accuracy report available. Run calculate_accuracy() first.")
            return
        
        report_content = f"""
ENHANCED BANK STATEMENT PARSER - ACCURACY REPORT

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

ENHANCEMENTS APPLIED:
- Advanced image preprocessing (4 variants per image)
- Multiple OCR configurations tested
- Enhanced resolution scaling (300 DPI)
- Skew correction
- Bilateral filtering for noise reduction
- CLAHE contrast enhancement
- Adaptive thresholding
- Multiple regex patterns for transaction parsing
- Enhanced date and amount parsing

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Enhanced accuracy report saved to: {output_path}")
        print(report_content)
    
    def save_parsed_results(self, output_path: str = "enhanced_parsed_results.csv"):
        """
        Save parsed results to CSV file.
        
        Args:
            output_path: Path to save the parsed results
        """
        if not self.parsed_results.empty:
            self.parsed_results.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Enhanced parsed results saved to: {output_path}")
        else:
            logger.warning("No enhanced parsed results to save")


def main():
    """
    Main function to run the enhanced bank statement parser.
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
    
    print("🚀 Starting Enhanced Bank Statement Parser...")
    
    # Initialize enhanced parser
    parser = EnhancedBankStatementParser(GROUND_TRUTH_PATH)
    
    # Parse all statements with enhanced processing
    print("📄 Enhanced parsing of bank statement images...")
    parsed_df = parser.parse_all_statements(IMAGES_FOLDER)
    
    # Save parsed results
    parser.save_parsed_results("enhanced_parsed_bank_statements.csv")
    
    # Calculate accuracy
    print("📊 Calculating enhanced accuracy metrics...")
    accuracy = parser.calculate_accuracy()
    
    # Generate report
    parser.generate_report("enhanced_bank_statement_parsing_report.txt")
    
    print("✅ Enhanced bank statement parsing completed!")
    print(f"📈 Enhanced Overall Accuracy: {accuracy.get('overall_accuracy', 0):.2f}%")
    print(f"📋 Enhanced Parsing: {accuracy.get('total_parsed_transactions', 0)} out of {accuracy.get('total_ground_truth_transactions', 0)} transactions")


if __name__ == "__main__":
    main()
