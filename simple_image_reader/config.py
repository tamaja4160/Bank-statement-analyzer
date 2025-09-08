"""Simple configuration for the image reader."""

from pathlib import Path

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

class SimpleConfig:
    """Simple configuration class for the image reader."""

    def __init__(self):
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
