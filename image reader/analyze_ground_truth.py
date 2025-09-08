import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_GROUND_TRUTH_PATH = "bank statement generator/bank_statements/ground_truth.csv"
SECTION_SEPARATOR_LENGTH = 50
STATISTICS_SEPARATOR_LENGTH = 30
SUMMARY_SEPARATOR_LENGTH = 20
EXAMPLE_STATEMENT_IDS = [1, 2, 3]

class GroundTruthAnalyzer:
    """Analyzes ground truth data for bank statement transactions."""

    def __init__(self, ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH):
        """
        Initialize the analyzer with the path to ground truth data.

        Args:
            ground_truth_path: Path to the CSV file containing ground truth data
        """
        self.ground_truth_path = Path(ground_truth_path)
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        """Lazy load ground truth data."""
        if self._data is None:
            self._data = self._load_ground_truth_data()
        return self._data

    def _load_ground_truth_data(self) -> pd.DataFrame:
        """Load ground truth data from CSV file."""
        try:
            if not self.ground_truth_path.exists():
                raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
            return pd.read_csv(self.ground_truth_path)
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            raise

    def get_total_transactions(self) -> int:
        """Get total number of transactions."""
        return len(self.data)

    def get_statement_counts(self) -> pd.Series:
        """Get transaction counts per statement."""
        return self.data['statement_id'].value_counts().sort_index()

    def get_summary_statistics(self) -> Dict[str, float]:
        """Calculate summary statistics for transactions per statement."""
        counts = self.get_statement_counts()
        return {
            'number_of_statements': len(counts),
            'average_transactions_per_statement': counts.mean(),
            'min_transactions_per_statement': counts.min(),
            'max_transactions_per_statement': counts.max()
        }

    def get_example_transactions(self, statement_ids: List[int] = None) -> Dict[int, pd.DataFrame]:
        """Get example transactions for specified statement IDs."""
        if statement_ids is None:
            statement_ids = EXAMPLE_STATEMENT_IDS

        examples = {}
        for stmt_id in statement_ids:
            stmt_transactions = self.data[self.data['statement_id'] == stmt_id]
            if not stmt_transactions.empty:
                examples[stmt_id] = stmt_transactions
        return examples

    def print_analysis_report(self) -> None:
        """Print comprehensive analysis report."""
        self._print_header()
        self._print_total_transactions()
        self._print_statement_counts()
        self._print_summary_statistics()
        self._print_example_transactions()

    def _print_header(self) -> None:
        """Print report header."""
        print("GROUND TRUTH ANALYSIS")
        print("=" * SECTION_SEPARATOR_LENGTH)

    def _print_total_transactions(self) -> None:
        """Print total transaction count."""
        total = self.get_total_transactions()
        print(f"Total transactions in ground truth: {total}")

    def _print_statement_counts(self) -> None:
        """Print transaction counts per statement."""
        counts = self.get_statement_counts()
        print("\nTransactions per statement:")
        print("-" * STATISTICS_SEPARATOR_LENGTH)

        for statement_id, count in counts.items():
            print(f"Statement {statement_id:3d}: {count} transactions")

    def _print_summary_statistics(self) -> None:
        """Print summary statistics."""
        stats = self.get_summary_statistics()
        print("\nSummary Statistics:")
        print("-" * SUMMARY_SEPARATOR_LENGTH)
        print(f"Number of statements: {stats['number_of_statements']}")
        print(f"Average transactions per statement: {stats['average_transactions_per_statement']:.2f}")
        print(f"Min transactions per statement: {stats['min_transactions_per_statement']}")
        print(f"Max transactions per statement: {stats['max_transactions_per_statement']}")

    def _print_example_transactions(self) -> None:
        """Print example transactions from first few statements."""
        examples = self.get_example_transactions()
        if not examples:
            return

        print("\nExample transactions from first few statements:")
        print("-" * SECTION_SEPARATOR_LENGTH)

        for stmt_id, transactions in examples.items():
            print(f"\nStatement {stmt_id} ({len(transactions)} transactions):")
            for _, trans in transactions.iterrows():
                print(f"  - {trans['date']}: {trans['amount']:.2f} EUR - {trans['identified_partner']}")


def analyze_ground_truth(ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH) -> None:
    """
    Analyze ground truth data and print comprehensive report.

    Args:
        ground_truth_path: Path to the ground truth CSV file
    """
    try:
        analyzer = GroundTruthAnalyzer(ground_truth_path)
        analyzer.print_analysis_report()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_ground_truth()
