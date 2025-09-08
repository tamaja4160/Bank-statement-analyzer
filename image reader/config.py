"""Configuration settings for the bank statement parser."""

from pathlib import Path
from typing import List

# File paths
DEFAULT_GROUND_TRUTH_PATH = "bank statement generator/bank_statements/ground_truth.csv"
DEFAULT_STATEMENTS_DIR = "bank statement generator/bank_statements"
DEFAULT_OUTPUT_CSV = "complete_parsed_transactions.csv"
DEFAULT_REPORT_TXT = "complete_parsing_report.txt"

# Tesseract configuration
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
OCR_CONFIGS = [
    '--oem 3 --psm 6 -l eng',
    '--oem 3 --psm 4 -l eng',
    '--oem 3 --psm 3 -l eng',
]

# Image processing
IMAGE_SCALE_FACTOR = 2
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2

# Transaction parsing patterns
TRANSACTION_PATTERNS = [
    # Pattern 1: Single line format - DD.MM. DD.MM. DESCRIPTION AMOUNT-
    r'(\d{1,2}\.\d{1,2})\.\s+(\d{1,2}\.\d{1,2})\.\s+(.+?)\s+(\d+[,\.]\d{1,2})-',

    # Pattern 2: Compact single line - DD.MM. DD.MM. DESCRIPTION AMOUNT-
    r'(\d{1,2}\.\d{1,2})\.\s*(\d{1,2}\.\d{1,2})\.\s*(.+?)\s+(\d+[,\.]\d{1,2})-',
]

# Transaction section markers
TRANSACTION_START_MARKER = 'KONTOSTAND'
TRANSACTION_END_MARKER = 'ABRECHNUNGSTERMIN'

# Partner identification patterns
PARTNER_PATTERNS = {
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

# Category classification keywords
CATEGORY_KEYWORDS = {
    'Shopping': ['ZALANDO', 'AMAZON', 'SHOPPING'],
    'Food': ['BURGER', 'RESTAURANT', 'FOOD'],
    'Telecommunications': ['VODAFONE', 'TELEFON', 'MOBILE'],
    'Fitness': ['BODYPOWER', 'GYM', 'FITNESS'],
    'Fuel': ['SHELL', 'JET', 'TANKSTELLE', 'FUEL'],
    'Insurance': ['ALLIANZ', 'INSURANCE', 'VERSICHERUNG'],
    'Utilities': ['STROM', 'STADTWERKE', 'UTILITIES']
}

# Report formatting
REPORT_TITLE = "COMPLETE BANK STATEMENT PARSING REPORT"
SECTION_SEPARATOR = "=" * 60
PER_STATEMENT_HEADER = "PER-STATEMENT RESULTS:"
RESULTS_SEPARATOR = "-" * 30

# Status indicators
STATUS_PERFECT = "✓ Perfect"
STATUS_PARTIAL = "~ Partial"
STATUS_ZERO = "✗ Zero"
STATUS_ERROR = "ERROR"

# Logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

class Config:
    """Central configuration class for the bank statement parser."""

    def __init__(self):
        # File paths - make them relative to the project root
        project_root = Path(__file__).parent.parent
        self.ground_truth_path = project_root / "bank statement generator" / "bank_statements" / "ground_truth.csv"
        self.statements_dir = project_root / "bank statement generator" / "bank_statements"
        self.output_csv = "complete_parsed_transactions.csv"
        self.report_txt = "complete_parsing_report.txt"

        # Tesseract settings
        self.tesseract_cmd = TESSERACT_CMD
        self.ocr_configs = OCR_CONFIGS.copy()

        # Image processing settings
        self.image_scale_factor = IMAGE_SCALE_FACTOR
        self.adaptive_thresh_block_size = ADAPTIVE_THRESH_BLOCK_SIZE
        self.adaptive_thresh_c = ADAPTIVE_THRESH_C

        # Transaction parsing settings
        self.transaction_patterns = TRANSACTION_PATTERNS.copy()
        self.transaction_start_marker = TRANSACTION_START_MARKER
        self.transaction_end_marker = TRANSACTION_END_MARKER

        # Classification settings
        self.partner_patterns = PARTNER_PATTERNS.copy()
        self.category_keywords = CATEGORY_KEYWORDS.copy()

        # Logging settings
        self.log_format = LOG_FORMAT
        self.log_level = LOG_LEVEL

        # Status constants
        self.status_perfect = STATUS_PERFECT
        self.status_partial = STATUS_PARTIAL
        self.status_zero = STATUS_ZERO
        self.status_error = STATUS_ERROR

        # Report formatting
        self.report_title = REPORT_TITLE
        self.section_separator = SECTION_SEPARATOR
        self.per_statement_header = PER_STATEMENT_HEADER
        self.results_separator = RESULTS_SEPARATOR

    def get_statement_image_path(self, statement_id: int) -> Path:
        """Get the file path for a specific statement image."""
        return self.statements_dir / f"statement_{statement_id}.png"

    def get_all_statement_files(self) -> List[Path]:
        """Get all PNG statement files sorted by ID."""
        png_files = list(self.statements_dir.glob("*.png"))
        png_files.sort(key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0)
        return png_files
