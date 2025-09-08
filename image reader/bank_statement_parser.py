"""Refactored bank statement parser with clean architecture."""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
import re

from config import Config
from ocr_processor import OCRProcessor
from transaction_parser import TransactionParser
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)

class BankStatementParser:
    """
    Main parser class that orchestrates the processing of bank statement images.

    This class follows clean architecture principles with dependency injection
    and separation of concerns.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the bank statement parser.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()
        self.ocr_processor = OCRProcessor(self.config)
        self.transaction_parser = TransactionParser(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.ground_truth_df = self._load_ground_truth()

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth data for comparison."""
        try:
            return pd.read_csv(self.config.ground_truth_path)
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            return pd.DataFrame()

    def process_single_statement(self, image_path: Path) -> Dict:
        """
        Process a single bank statement image.

        Args:
            image_path: Path to the statement image

        Returns:
            Dictionary containing processing results for this statement
        """
        try:
            # Extract statement ID from filename
            statement_id = self._extract_statement_id(image_path)

            # Extract text from image
            ocr_text, confidence = self.ocr_processor.extract_text_from_image(image_path)

            # Parse transactions
            transactions = self.transaction_parser.parse_transactions(ocr_text, statement_id)

            # Get ground truth count for comparison
            gt_count = self._get_ground_truth_count(statement_id)

            # Determine status
            status = self.report_generator.determine_extraction_status(len(transactions), gt_count)

            return {
                'statement_id': statement_id,
                'transactions': transactions,
                'ground_truth_count': gt_count,
                'extracted_count': len(transactions),
                'confidence': confidence,
                'status': status,
                'image_path': image_path
            }

        except Exception as e:
            logger.error(f"Failed to process statement {image_path}: {e}")
            statement_id = self._extract_statement_id(image_path)
            return {
                'statement_id': statement_id,
                'transactions': [],
                'ground_truth_count': self._get_ground_truth_count(statement_id),
                'extracted_count': 0,
                'confidence': 0.0,
                'status': self.config.status_error,
                'image_path': image_path
            }

    def process_all_statements(self, statements_dir: Optional[Path] = None) -> Dict:
        """
        Process all bank statement images in the specified directory.

        Args:
            statements_dir: Directory containing statement images. If None, uses config default.

        Returns:
            Comprehensive results dictionary
        """
        statements_dir = statements_dir or self.config.statements_dir

        if not statements_dir.exists():
            raise FileNotFoundError(f"Statements directory not found: {statements_dir}")

        # Get all statement files
        statement_files = self.config.get_all_statement_files()
        if not statement_files:
            logger.warning(f"No statement files found in {statements_dir}")
            return self._create_empty_results()

        logger.info(f"Processing {len(statement_files)} bank statements")

        # Process each statement
        all_results = {}
        all_transactions = []
        total_confidence = 0
        processed_count = 0

        # Counters for performance metrics
        perfect_matches = 0
        partial_matches = 0
        zero_extractions = 0

        # Print processing header
        self.report_generator.print_processing_summary({
            'summary': {'total_statements': len(statement_files)},
            'results': {}
        })

        for image_path in statement_files:
            result = self.process_single_statement(image_path)

            statement_id = result['statement_id']
            all_results[statement_id] = {
                'ground_truth': result['ground_truth_count'],
                'extracted': result['extracted_count'],
                'confidence': result['confidence'],
                'status': result['status']
            }

            all_transactions.extend(result['transactions'])
            total_confidence += result['confidence']
            processed_count += 1

            # Update counters
            if result['status'] == self.config.status_perfect:
                perfect_matches += 1
            elif result['status'] == self.config.status_partial:
                partial_matches += 1
            elif result['status'] == self.config.status_zero:
                zero_extractions += 1

            # Print progress for this statement
            print(f"{statement_id:<12} {result['ground_truth_count']:<4} "
                  f"{result['extracted_count']:<4} {result['confidence']:<6.1f} {result['status']}")

        # Calculate summary statistics
        total_gt = sum(r['ground_truth'] for r in all_results.values())
        total_extracted = len(all_transactions)
        avg_confidence = total_confidence / processed_count if processed_count > 0 else 0

        summary = {
            'total_statements': len(all_results),
            'total_gt': total_gt,
            'total_extracted': total_extracted,
            'extraction_rate': total_extracted / total_gt if total_gt > 0 else 0,
            'avg_confidence': avg_confidence,
            'perfect_matches': perfect_matches,
            'partial_matches': partial_matches,
            'zero_extractions': zero_extractions
        }

        results = {
            'results': all_results,
            'transactions': all_transactions,
            'summary': summary
        }

        # Print final summary
        self.report_generator.print_processing_summary(results)

        return results

    def _extract_statement_id(self, image_path: Path) -> int:
        """Extract statement ID from image filename."""
        match = re.search(r'statement_(\d+)\.png', image_path.name)
        if match:
            return int(match.group(1))
        else:
            # Fallback: try to extract any number from filename
            match = re.search(r'(\d+)', image_path.name)
            return int(match.group(1)) if match else 0

    def _get_ground_truth_count(self, statement_id: int) -> int:
        """Get the number of ground truth transactions for a statement."""
        if self.ground_truth_df.empty:
            return 0
        statement_data = self.ground_truth_df[self.ground_truth_df['statement_id'] == statement_id]
        return len(statement_data)

    def _create_empty_results(self) -> Dict:
        """Create empty results structure."""
        return {
            'results': {},
            'transactions': [],
            'summary': {
                'total_statements': 0,
                'total_gt': 0,
                'total_extracted': 0,
                'extraction_rate': 0,
                'avg_confidence': 0,
                'perfect_matches': 0,
                'partial_matches': 0,
                'zero_extractions': 0
            }
        }

    def save_results(self, results: Dict, csv_path: Optional[Path] = None,
                    report_path: Optional[Path] = None) -> bool:
        """
        Save processing results to files.

        Args:
            results: Results dictionary from processing
            csv_path: Optional custom path for CSV output
            report_path: Optional custom path for report output

        Returns:
            True if both saves were successful
        """
        csv_success = self.report_generator.generate_csv_output(
            results.get('transactions', []), csv_path
        )

        report_success = self.report_generator.generate_processing_report(
            results, report_path
        )

        return csv_success and report_success
