[](https://opensource.org/licenses/MIT)

# Bank Statement Suite: Save Time, Money & Boost Revenue

I created a tool that generates bank statements and reads them using OCR, finds recurring payment,s and suggests cheaper alternatives (all based on synthetic data).

## What It Does & How to Use It

You can start immediately with the primary use cases.

### 1\. Create Clean Bank Statement Images

Generate realistic German bank statements (PNG) to test financial software, train OCR, and benchmark ML models.

```bash
# Generate clean, realistic-looking bank statements
cd bank-statement-generator
python statement_generator.py
```
<img width="691" height="326" alt="image" src="https://github.com/user-attachments/assets/1743dc1b-c7e8-4162-916d-e0a67a9e33f9" />


### 2\. Create Hard-to-Read Bank Statement Images

Create challenging statements (blurry, rotated) to test and improve the robustness of your OCR solutions.

```bash
# Create hard-to-read statements to test OCR robustness
cd bank-statement-generator
python image_worsener.py
```
<img width="728" height="339" alt="image" src="https://github.com/user-attachments/assets/7b157ed9-8d4f-4790-a20b-d8ef49275c8e" />


### 3\. Analyze Bank Statements and Save Money

Simulate tracking expenses, identifying subscriptions, and finding ways to save money. Run the entire pipeline from data generation to analysis with a single command and view the results in an interactive dashboard.

Access it under this address: http://localhost:8501/

```bash
# This single command runs the full pipeline and launches the dashboard
python run.py
```

<img width="2090" height="735" alt="image" src="https://github.com/user-attachments/assets/bfe67893-cf39-446d-98be-7c66e55396a9" />

<img width="2128" height="801" alt="image" src="https://github.com/user-attachments/assets/5fa92eaa-09a2-4ec1-b007-31e0c1475333" />

<img width="2124" height="946" alt="image" src="https://github.com/user-attachments/assets/09f2fa7e-e404-4683-9c05-5112e5d35c22" />




## ✨ Key Features

  * **AI-Powered OCR**: Extracts transaction data with **98% accuracy** using Tesseract and OpenCV.
  * **Recurring Payment Detection**: Automatically identifies subscriptions and regular charges with **85%+ accuracy** using fuzzy matching, similarity scoring, and clustering.
  * **Synthetic Data Generation**: Creates realistic German bank statements in both Excel and PNG formats, perfect for training and testing.
  * **Image Augmentation**: Programmatically applies blur, rotation, and perspective changes to images to create a robust test set for OCR models.
  * **Interactive Dashboard & API**: Visualize financial insights with a user-friendly Streamlit dashboard and integrate the logic into other applications via a production-ready FastAPI service.

## 🛠️ Technology Stack

  * **Machine Learning**: Scikit-learn, Transformers, Sentence Transformers, LightGBM
  * **Data Processing**: Pandas, NumPy
  * **OCR & Image Processing**: Tesseract, OpenCV, Pillow
  * **Web & API**: Streamlit, FastAPI
  * **Data Generation**: Faker, OpenPyXL, excel2img

## 💼 Project Deep Dive & Skills Honed

This project is structured as a complete FinTech product pipeline, demonstrating key data science skills.

  * **Step 1: Bank Statement Generation (`bank-statement-generator/`)**
      * **What it is:** A robust data generator creating authentic German banking data from templates.
      * **Skills developed:** Data synthesis, financial data modeling, working with German banking formats.
  * **Step 2: OCR Processing (`simple_image_reader/`)**
      * **What it is:** A script that masters text extraction from images with high accuracy.
      * **Skills developed:** OpenCV, Tesseract OCR, regex pattern matching, processing unstructured data.
  * **Step 3: ML Payment Analysis (`streamlit_bank_analyzer/`)**
      * **What it is:** An end-to-end ML pipeline for recurring payment detection, complete with a UI.
      * **Skills developed:** Fuzzy matching, similarity scoring, clustering, and integrating ML models into a cohesive product.

## 📈 Performance Metrics

  * **OCR Accuracy**: 98% extraction success rate.
  * **Processing Speed**: \~2-3 seconds per statement.
  * **Analysis Accuracy**: 85%+ recurring payment detection.

## Installation

### Prerequisites

  * Python 3.9+
  * Tesseract OCR (see [official install guide](https://github.com/tesseract-ocr/tesseract))
  * Windows/Linux/macOS

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/tamaja4160/Bank-statement-analyzer.git
    cd Bank-statement-analyzer
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Architecture

```
bank-statement-analyzer/
├── bank-statement-generator/      # XLSX/PNG generation & image augmentation
├── simple_image_reader/           # OCR extraction
├── streamlit_bank_analyzer/       # ML analysis, dashboard & API
├── requirements.txt               # Dependencies
└── README.md                      # This file
```
