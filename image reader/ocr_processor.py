"""OCR processing module for bank statement images."""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing of bank statement images."""

    def __init__(self, config: Config):
        """
        Initialize the OCR processor.

        Args:
            config: Configuration object containing OCR settings
        """
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def preprocess_image(self, image_path: Path) -> List[np.ndarray]:
        """
        Create multiple preprocessed variants of an image for OCR.

        Args:
            image_path: Path to the image file

        Returns:
            List of preprocessed image variants

        Raises:
            ValueError: If the image cannot be loaded
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale to higher resolution for better OCR
        scale_factor = self.config.image_scale_factor
        height, width = gray.shape
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor),
                         interpolation=cv2.INTER_CUBIC)

        variants = []

        # Variant 1: Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_thresh_block_size,
            self.config.adaptive_thresh_c
        )
        variants.append(adaptive_thresh)

        # Variant 2: OTSU threshold
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu_thresh)

        return variants

    def extract_text_from_image(self, image_path: Path) -> Tuple[str, float]:
        """
        Extract text from an image using multiple OCR configurations.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Create preprocessed image variants
            image_variants = self.preprocess_image(image_path)

            # Try multiple OCR configurations
            best_text = ""
            best_confidence = 0.0

            for variant in image_variants:
                for config in self.config.ocr_configs:
                    try:
                        # Get OCR result with confidence data
                        data = pytesseract.image_to_data(
                            variant,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )

                        # Calculate average confidence
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        if confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                            if avg_confidence > best_confidence:
                                best_confidence = avg_confidence
                                best_text = pytesseract.image_to_string(variant, config=config)

                    except Exception as e:
                        logger.warning(f"OCR configuration failed: {config}, error: {e}")
                        continue

            return best_text, best_confidence

        except Exception as e:
            logger.error(f"Failed to extract text from {image_path}: {e}")
            return "", 0.0

    def extract_text_with_fallback(self, image_path: Path) -> str:
        """
        Extract text with fallback to simpler OCR if advanced fails.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text string
        """
        text, confidence = self.extract_text_from_image(image_path)

        # If confidence is too low, try a simpler approach
        if confidence < 10:  # Arbitrary threshold
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config='--oem 1 --psm 6')
            except Exception as e:
                logger.warning(f"Fallback OCR also failed: {e}")

        return text
