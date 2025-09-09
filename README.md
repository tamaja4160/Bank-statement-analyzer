# Bank Statement Suite

A comprehensive suite of tools for generating, processing, and analyzing bank statements using OCR and machine learning.

## Overview

This repository provides an end-to-end solution for:
- Generating realistic bank statements
- Applying image augmentations to test OCR robustness
- Extracting transaction data using OCR
- Analyzing recurring payments and spending patterns

## Features

- **Bank Statement Generation**: Create authentic-looking German bank statements with varied transactions
- **Image Degradation**: Apply random augmentations (rotation, blur, brightness changes) to test OCR systems
- **OCR Processing**: Extract transaction data from images with high accuracy (96% success rate)
- **Payment Analysis**: Identify recurring payments using fuzzy matching and pattern recognition
- **Web Interface**: Streamlit app for interactive analysis
- **REST API**: FastAPI service for programmatic integration

## Projects

### 🏦 Bank Statement Generator (`bank-statement-generator/`)
Generate synthetic bank statements from templates, including:
- Realistic German banking data
- Mix of recurring and one-off payments
- Excel and PNG output formats
- Ground truth CSV for testing

### 🔍 Image Worsener (`image-worsener/`)
Apply augmentation techniques to make images harder to read:
- Perspective transforms
- Rotation and blur
- Brightness/contrast adjustments
- Useful for testing OCR accuracy under challenging conditions

### 📷 Simple Image Reader (`simple_image_reader/`)
High-accuracy OCR processor for bank statement images:
- Multiple OCR strategies with OpenCV preprocessing
- Transaction extraction with regex patterns
- CSV/JSON output formats
- 96% extraction success rate

### 📊 Streamlit Bank Analyzer (`streamlit_bank_analyzer/`)
Complete web application for bank statement analysis:
- Integrated statement generation and OCR processing
- Recurring payment detection with ML algorithms
- Financial insights and pattern analysis
- REST API with FastAPI for external integrations

## Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR (download from [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract))
- Windows/Linux/macOS

### Setup
1. Clone the repository:
```bash
git clone https://github.com/tamaja4160/Projects.git
cd Projects
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR system-wide

## Usage

Each project can be run independently. See individual README files for detailed instructions.

### Quick Start
1. Generate sample statements:
```bash
cd bank-statement-generator
python statement_generator.py
```

2. Extract transactions via OCR:
```bash
cd simple_image_reader
python main.py ../bank-statement-generator/bank_statements/
```

3. Run the full analysis app:
```bash
cd streamlit_bank_analyzer
streamlit run app.py
```

## Technology Stack

- **OCR**: Tesseract with OpenCV preprocessing
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Transformers, Sentence Transformers
- **Image Processing**: Pillow, OpenCV
- **Web Framework**: Streamlit, FastAPI
- **Excel Processing**: OpenPyXL, excel2img
- **Data Generation**: Faker

## Architecture

```
bank-statement-suite/
├── bank-statement-generator/     # XLSX/PNG generation
├── image-worsener/              # Image augmentation
├── simple_image_reader/         # OCR extraction
├── streamlit_bank_analyzer/     # Full analysis app
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Performance Metrics

- **OCR Accuracy**: 96% extraction success rate (Simple Image Reader)
- **Processing Speed**: ~2-3 seconds per statement
- **Analysis Accuracy**: 85%+ recurring payment detection
- **Success Rate**: 80%+ overall workflow completion

## Examples

### Generated Statement
```
KONTOSTAND AM 15.03.2024
AB 15.03.2024 BIS 20.03.2024

Beleg Datum     Valuta Datum    Buchungstext                  Betrag
----------------------------------------------------------------------
15.03.          16.03.         VODAFONE GMBH                  29,99-
17.03.          18.03.         ALLIANZ SE                     89,50-
19.03.          20.03.         transfer 042023                150,00-

KONTOSTAND AM 20.03.2024                                       269,49-
```

### OCR Extraction Results
```json
{
  "transactions": [
    {
      "date1": "15.03",
      "date2": "16.03",
      "formatted_date": "15.03.2024",
      "amount_str": "29,99-",
      "amount_value": -29.99,
      "description": "VODAFONE GMBH",
      "raw_line": "15.03. 16.03. VODAFONE GMBH 29,99-"
    }
  ],
  "extracted_count": 3,
  "oRC_confidence": 94.2
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- Multi-language bank statement support
- Advanced ML models for better pattern recognition
- API integrations with real banking systems
- Batch processing capabilities
- Export functionality for accounting software
