import cv2
import pytesseract
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr_on_single_image():
    """Test OCR on a single image to debug the issue"""
    image_path = "bank statement generator/bank_statements/statement_1.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image loaded successfully: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try basic OCR
    try:
        text = pytesseract.image_to_string(gray)
        print(f"OCR Text length: {len(text)}")
        print(f"OCR Text (first 200 chars): {text[:200]}")
        
        # Try with confidence
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average confidence: {avg_confidence:.2f}")
        else:
            print("No confidence data available")
            
    except Exception as e:
        print(f"OCR Error: {e}")

if __name__ == "__main__":
    test_ocr_on_single_image()
