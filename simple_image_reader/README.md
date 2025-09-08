# Simple Image Reader

A lightweight tool for extracting transaction lines from bank statement images using OCR. This tool focuses purely on extraction without ground truth comparison or transaction categorization.

## Features

- **Pure Extraction**: Extracts raw transaction lines from bank statement images
- **Multiple OCR Strategies**: Uses different OCR configurations for best results
- **Image Preprocessing**: Applies multiple image processing techniques to improve OCR accuracy
- **Flexible Input**: Process single images or entire directories
- **Multiple Output Formats**: Save results as CSV or JSON
- **No Dependencies on Ground Truth**: Works independently without requiring reference data

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - Download and install from: https://github.com/tesseract-ocr/tesseract
   - Make sure it's installed at: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Or update the path in `config.py`

## Usage

### Command Line

Process a single image:
```bash
python main.py path/to/image.png
```

Process all images in a directory:
```bash
python main.py path/to/directory
```

Process first N images in a directory:
```bash
python main.py path/to/directory 5
```

Process default bank statements directory:
```bash
python main.py
```

### Programmatic Usage

```python
from pathlib import Path
from simple_image_reader import SimpleImageReader

# Initialize reader
reader = SimpleImageReader()

# Process single image
result = reader.process_single_image(Path("statement.png"))
print(f"Found {result['extracted_count']} transactions")

# Process directory
results = reader.process_directory(Path("statements/"))
print(f"Total transactions: {len(results['transactions'])}")

# Save results
reader.save_results_to_csv(results['transactions'], "output.csv")
reader.save_results_to_json(results, "results.json")
```

## Output Format

### Transaction Data
Each extracted transaction contains:
- `date1`: First date (DD.MM format)
- `date2`: Second date (DD.MM format)  
- `formatted_date`: Full date (DD.MM.YYYY format)
- `amount_str`: Original amount string
- `amount_value`: Parsed amount as float
- `description`: Transaction description
- `raw_line`: Reconstructed transaction line

### CSV Output
Transactions are saved to CSV with all fields as columns.

### JSON Output
Complete results including:
- Summary statistics
- Individual image results
- All extracted transactions
- OCR confidence scores

## Configuration

Edit `config.py` to customize:
- Tesseract path and OCR configurations
- Image processing parameters
- Transaction parsing patterns
- Section markers for transaction blocks

## File Structure

```
simple_image_reader/
├── __init__.py
├── config.py                 # Configuration settings
├── ocr_processor.py          # OCR processing logic
├── transaction_extractor.py  # Transaction extraction logic
├── simple_image_reader.py    # Main reader class
├── main.py                   # Command line interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Example Output

```
Processing 5 images...
============================================================
Image                          Confidence   Transactions Status
------------------------------------------------------------
statement_1.png                85.2         3            ✓ Success
statement_2.png                78.9         2            ✓ Success
statement_3.png                92.1         4            ✓ Success
statement_4.png                0.0          0            ✗ Failed
statement_5.png                88.5         3            ✓ Success

============================================================
EXTRACTION SUMMARY
============================================================
Total images processed: 5
Successful extractions: 4
Failed extractions: 1
Total transactions extracted: 12
Average OCR confidence: 86.2%
Success rate: 80.0%
```

## Limitations

- Requires Tesseract OCR to be installed
- Optimized for German bank statement format
- May require adjustment of regex patterns for different statement formats
- OCR accuracy depends on image quality
