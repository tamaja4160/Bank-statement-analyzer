import os
import cv2
import numpy as np
import pytesseract

# Configure Tesseract path and environment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess the bank statement image for better OCR results."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def test_ocr_on_sample():
    """Test OCR on a sample bank statement image."""
    image_path = "bank statement generator/bank_statements/statement_1.png"
    
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Extract text using Tesseract
    ocr_config = '--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(processed_image, config=ocr_config)
    
    print("=== OCR OUTPUT FOR STATEMENT 1 ===")
    print(text)
    print("=== END OCR OUTPUT ===")
    
    # Show each line with line numbers
    print("\n=== LINE BY LINE ANALYSIS ===")
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip():
            print(f"Line {i:2d}: '{line}'")
    
    print(f"\nTotal lines: {len(lines)}")
    print(f"Non-empty lines: {len([l for l in lines if l.strip()])}")

if __name__ == "__main__":
    test_ocr_on_sample()
