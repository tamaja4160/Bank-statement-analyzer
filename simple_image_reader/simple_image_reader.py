"""Simple image reader that extracts transaction lines from bank statement images."""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging
import json

from config import SimpleConfig
from ocr_processor import SimpleOCRProcessor
from transaction_extractor import SimpleTransactionExtractor

logger = logging.getLogger(__name__)

class SimpleImageReader:
    """
    Simple image reader that extracts raw transaction lines from bank statement images.
    No ground truth comparison or categorization - just pure extraction.
    """

    def __init__(self, config: Optional[SimpleConfig] = None):
        """Initialize the simple image reader."""
        self.config = config or SimpleConfig()
        self.ocr_processor = SimpleOCRProcessor(self.config)
        self.transaction_extractor = SimpleTransactionExtractor(self.config)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_single_image(self, image_path: Path) -> Dict:
        """
        Process a single bank statement image and extract transaction lines.
        
        Args:
            image_path: Path to the statement image
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Extract text from image
            ocr_text, confidence = self.ocr_processor.extract_text_from_image(image_path)
            
            # Extract transactions
            transactions = self.transaction_extractor.extract_transactions(ocr_text)
            
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'ocr_confidence': confidence,
                'extracted_count': len(transactions),
                'transactions': transactions,
                'raw_ocr_text': ocr_text,
                'success': True
            }

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'ocr_confidence': 0.0,
                'extracted_count': 0,
                'transactions': [],
                'raw_ocr_text': "",
                'success': False,
                'error': str(e)
            }

    def process_directory(self, directory_path: Path, limit: Optional[int] = None) -> Dict:
        """
        Process images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            limit: Maximum number of images to process (None for all)
            
        Returns:
            Dictionary containing all results
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Get all image files (PNG and JPG)
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(directory_path.glob(ext))
        
        if not image_files:
            logger.warning(f"No image files found in {directory_path}")
            return self._create_empty_results()

        # Sort files and apply limit if specified
        image_files = sorted(image_files)
        if limit is not None and limit > 0:
            image_files = image_files[:limit]
            logger.info(f"Found {len(image_files)} images, processing first {limit} (limit applied)")
        else:
            logger.info(f"Found {len(image_files)} images to process")
        
        all_results = []
        all_transactions = []
        total_confidence = 0
        successful_extractions = 0

        print(f"\nProcessing {len(image_files)} images...")
        print("=" * 60)
        print(f"{'Image':<30} {'Confidence':<12} {'Transactions':<12} {'Status'}")
        print("-" * 60)

        for image_path in sorted(image_files):
            result = self.process_single_image(image_path)
            all_results.append(result)
            
            if result['success']:
                all_transactions.extend(result['transactions'])
                total_confidence += result['ocr_confidence']
                successful_extractions += 1
                status = "✓ Success"
            else:
                status = "✗ Failed"
            
            # Print progress
            print(f"{result['image_name']:<30} {result['ocr_confidence']:<12.1f} "
                  f"{result['extracted_count']:<12} {status}")

        # Calculate summary
        avg_confidence = total_confidence / successful_extractions if successful_extractions > 0 else 0
        
        summary = {
            'total_images': len(image_files),
            'successful_extractions': successful_extractions,
            'failed_extractions': len(image_files) - successful_extractions,
            'total_transactions': len(all_transactions),
            'average_confidence': avg_confidence,
            'success_rate': successful_extractions / len(image_files) if image_files else 0
        }

        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {summary['total_images']}")
        print(f"Successful extractions: {summary['successful_extractions']}")
        print(f"Failed extractions: {summary['failed_extractions']}")
        print(f"Total transactions extracted: {summary['total_transactions']}")
        print(f"Average OCR confidence: {summary['average_confidence']:.1f}%")
        print(f"Success rate: {summary['success_rate']:.1%}")

        return {
            'summary': summary,
            'results': all_results,
            'transactions': all_transactions
        }

    def _create_empty_results(self) -> Dict:
        """Create empty results structure."""
        return {
            'summary': {
                'total_images': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_transactions': 0,
                'average_confidence': 0,
                'success_rate': 0
            },
            'results': [],
            'transactions': []
        }

    def save_results_to_csv(self, transactions: List[Dict], output_path: str = "extracted_transactions.csv") -> bool:
        """
        Save extracted transactions to CSV file.
        
        Args:
            transactions: List of transaction dictionaries
            output_path: Path for output CSV file
            
        Returns:
            True if successful
        """
        try:
            if not transactions:
                logger.warning("No transactions to save")
                return False

            df = pd.DataFrame(transactions)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(transactions)} transactions to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return False

    def save_results_to_json(self, results: Dict, output_path: str = "extraction_results.json") -> bool:
        """
        Save complete results to JSON file.
        
        Args:
            results: Complete results dictionary
            output_path: Path for output JSON file
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved complete results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False
