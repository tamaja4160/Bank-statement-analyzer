"""Main entry point for the refactored bank statement parser."""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bank_statement_parser import BankStatementParser
from config import Config

logger = logging.getLogger(__name__)

def main():
    """Main function to process all bank statements."""
    try:
        # Initialize parser with default configuration
        parser = BankStatementParser()

        # Process all statements
        results = parser.process_all_statements()

        # Save results
        success = parser.save_results(results)

        if success:
            logger.info("Processing completed successfully")
            print(f"\nSaved {len(results['transactions'])} transactions to CSV")
            print("Detailed report saved to text file")
        else:
            logger.warning("Some files may not have been saved successfully")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
