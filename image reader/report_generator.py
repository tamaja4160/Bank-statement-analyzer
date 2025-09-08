"""Report generation module for bank statement processing results."""

import pandas as pd
from typing import Dict, List
from pathlib import Path
import logging

from config import Config

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Handles generation of processing reports and CSV outputs."""

    def __init__(self, config: Config):
        """
        Initialize the report generator.

        Args:
            config: Configuration object containing output settings
        """
        self.config = config

    def generate_csv_output(self, transactions: List[Dict], output_path: Path = None) -> bool:
        """
        Generate CSV file from parsed transactions.

        Args:
            transactions: List of transaction dictionaries
            output_path: Optional custom output path

        Returns:
            True if successful, False otherwise
        """
        if output_path is None:
            output_path = Path(self.config.output_csv)

        try:
            if not transactions:
                logger.warning("No transactions to save to CSV")
                return False

            df = pd.DataFrame(transactions)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(transactions)} transactions to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save CSV output: {e}")
            return False

    def generate_processing_report(self, results: Dict, report_path: Path = None) -> bool:
        """
        Generate detailed processing report.

        Args:
            results: Dictionary containing processing results and summary
            report_path: Optional custom report path

        Returns:
            True if successful, False otherwise
        """
        if report_path is None:
            report_path = Path(self.config.report_txt)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                self._write_report_header(f)
                self._write_summary_section(f, results.get('summary', {}))
                self._write_detailed_results(f, results.get('results', {}))

            logger.info(f"Detailed report saved to {report_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False

    def _write_report_header(self, file) -> None:
        """Write report header."""
        file.write(f"{self.config.report_title}\n")
        file.write(f"{self.config.section_separator}\n\n")

    def _write_summary_section(self, file, summary: Dict) -> None:
        """Write summary statistics section."""
        file.write("SUMMARY STATISTICS\n")
        file.write(f"{self.config.results_separator}\n")
        file.write(f"Total statements processed: {summary.get('total_statements', 0)}\n")
        file.write(f"Ground truth transactions: {summary.get('total_gt', 0)}\n")
        file.write(f"Extracted transactions: {summary.get('total_extracted', 0)}\n")
        file.write(f"Overall extraction rate: {summary.get('extraction_rate', 0):.1%}\n")
        file.write(f"Average OCR confidence: {summary.get('avg_confidence', 0):.1f}%\n\n")

        # Performance metrics
        perfect_matches = summary.get('perfect_matches', 0)
        partial_matches = summary.get('partial_matches', 0)
        zero_extractions = summary.get('zero_extractions', 0)
        total_statements = summary.get('total_statements', 1)  # Avoid division by zero

        file.write("PERFORMANCE METRICS\n")
        file.write(f"{self.config.results_separator}\n")
        file.write(f"Perfect matches (100%): {perfect_matches} statements ({perfect_matches/total_statements:.1%})\n")
        file.write(f"Partial matches (>0%):  {partial_matches} statements ({partial_matches/total_statements:.1%})\n")
        file.write(f"Zero extractions (0%):  {zero_extractions} statements ({zero_extractions/total_statements:.1%})\n")
        file.write(f"Statement-level success: {(perfect_matches + partial_matches)/total_statements:.1%}\n\n")

    def _write_detailed_results(self, file, results: Dict) -> None:
        """Write per-statement detailed results."""
        file.write(f"{self.config.per_statement_header}\n")
        file.write(f"{self.config.results_separator}\n")

        for stmt_id in sorted(results.keys()):
            result = results[stmt_id]
            gt_count = result.get('ground_truth', 0)
            extracted_count = result.get('extracted', 0)
            confidence = result.get('confidence', 0)
            status = result.get('status', 'UNKNOWN')

            file.write(f"Statement {stmt_id}: {extracted_count}/{gt_count} "
                      f"({confidence:.1f}% conf) - {status}\n")

    def print_processing_summary(self, results: Dict) -> None:
        """
        Print processing summary to console.

        Args:
            results: Dictionary containing processing results
        """
        summary = results.get('summary', {})
        per_statement_results = results.get('results', {})

        print(f"\nPROCESSING ALL {summary.get('total_statements', 0)} BANK STATEMENTS")
        print(f"{self.config.section_separator}")

        # Header
        print("Statement ID".ljust(12), "GT".ljust(4), "Ext".ljust(4), "Conf%".ljust(6), "Status")
        print("-" * 45)

        # Per-statement results
        for stmt_id in sorted(per_statement_results.keys()):
            result = per_statement_results[stmt_id]
            gt_count = result.get('ground_truth', 0)
            extracted_count = result.get('extracted', 0)
            confidence = result.get('confidence', 0)
            status = result.get('status', 'UNKNOWN')

            print(f"{stmt_id:<12} {gt_count:<4} {extracted_count:<4} {confidence:<6.1f} {status}")

        # Overall summary
        print(f"\n{self.config.section_separator}")
        print("COMPREHENSIVE RESULTS SUMMARY")
        print(f"{self.config.section_separator}")
        print(f"Total statements processed: {summary.get('total_statements', 0)}")
        print(f"Ground truth transactions: {summary.get('total_gt', 0)}")
        print(f"Extracted transactions: {summary.get('total_extracted', 0)}")
        print(f"Overall extraction rate: {summary.get('extraction_rate', 0):.1%}")
        print(f"Average OCR confidence: {summary.get('avg_confidence', 0):.1f}%")

        # Performance breakdown
        perfect_matches = summary.get('perfect_matches', 0)
        partial_matches = summary.get('partial_matches', 0)
        zero_extractions = summary.get('zero_extractions', 0)
        total_statements = summary.get('total_statements', 1)

        print()
        print(f"Perfect matches (100%): {perfect_matches} statements ({perfect_matches/total_statements:.1%})")
        print(f"Partial matches (>0%):  {partial_matches} statements ({partial_matches/total_statements:.1%})")
        print(f"Zero extractions (0%):  {zero_extractions} statements ({zero_extractions/total_statements:.1%})")
        print(f"Statement-level success: {(perfect_matches + partial_matches)/total_statements:.1%}")

    def determine_extraction_status(self, extracted_count: int, ground_truth_count: int) -> str:
        """
        Determine the extraction status based on counts.

        Args:
            extracted_count: Number of transactions extracted
            ground_truth_count: Number of ground truth transactions

        Returns:
            Status string
        """
        if extracted_count == ground_truth_count:
            return self.config.status_perfect
        elif extracted_count > 0:
            return self.config.status_partial
        else:
            return self.config.status_zero
