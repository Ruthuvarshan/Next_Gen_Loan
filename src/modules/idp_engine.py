"""
Module 1: Intelligent Document Processing (IDP) Engine

This module handles the extraction of structured data from unstructured 
financial documents (pay stubs, bank statements) using:
1. OpenCV for image preprocessing
2. Tesseract for OCR
3. spaCy for Named Entity Recognition (NER)
"""

import cv2
import numpy as np
import pytesseract
import spacy
import pdfplumber
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Union, List
from spacy.matcher import Matcher
import re

from src.utils.config import settings


class IDPEngine:
    """
    Intelligent Document Processing engine for extracting structured data
    from financial documents.
    """
    
    def __init__(self):
        """Initialize the IDP engine with necessary models and configurations."""
        # Set Tesseract path
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(settings.spacy_model)
        except OSError:
            # Fallback to basic model if custom model not available
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize spaCy Matcher for rule-based extraction
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Define spaCy Matcher patterns for common financial document fields."""
        
        # Pattern for Net Pay
        net_pay_pattern = [
            {"LOWER": {"IN": ["net", "take-home"]}},
            {"LOWER": {"IN": ["pay", "income", "salary"]}, "OP": "?"},
            {"TEXT": {"REGEX": r"[:=]?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"\$?\d+[,\d]*\.?\d*"}}
        ]
        self.matcher.add("NET_PAY", [net_pay_pattern])
        
        # Pattern for Gross Pay
        gross_pay_pattern = [
            {"LOWER": "gross"},
            {"LOWER": {"IN": ["pay", "income", "salary"]}, "OP": "?"},
            {"TEXT": {"REGEX": r"[:=]?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"\$?\d+[,\d]*\.?\d*"}}
        ]
        self.matcher.add("GROSS_PAY", [gross_pay_pattern])
        
        # Pattern for Pay Period
        pay_period_pattern = [
            {"LOWER": "pay"},
            {"LOWER": "period"},
            {"TEXT": {"REGEX": r"[:=]?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{2,4}"}}
        ]
        self.matcher.add("PAY_PERIOD", [pay_period_pattern])
        
        # Pattern for Account Balance
        balance_pattern = [
            {"LOWER": {"IN": ["balance", "ending", "current"]}},
            {"LOWER": {"IN": ["balance", "amount"]}, "OP": "?"},
            {"TEXT": {"REGEX": r"[:=]?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"\$?\d+[,\d]*\.?\d*"}}
        ]
        self.matcher.add("ACCOUNT_BALANCE", [balance_pattern])
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply OpenCV preprocessing to improve OCR accuracy.
        
        Steps:
        1. Grayscale conversion
        2. Denoising (Gaussian blur)
        3. Adaptive thresholding (binarization)
        4. Deskewing
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise with Gaussian blur
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Deskew the image
        deskewed = self._deskew(binary)
        
        return deskewed
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in the image.
        
        Args:
            image: Binary image
            
        Returns:
            Deskewed image
        """
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(image > 0))
        
        # Calculate minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image to deskew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF file.
        First tries direct text extraction; if that fails, uses OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            # Try direct text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If direct extraction yielded substantial text, return it
            if len(text.strip()) > 50:
                return text
        except Exception as e:
            print(f"Direct PDF text extraction failed: {e}")
        
        # If direct extraction failed, use OCR
        return self.extract_text_from_image(pdf_path)
    
    def extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image file using OCR.
        
        Args:
            image_path: Path to image file (or PDF to be treated as image)
            
        Returns:
            Extracted text
        """
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(preprocessed)
        
        return text
    
    def extract_structured_data(self, text: str, document_type: str = "paystub") -> Dict:
        """
        Extract structured data from raw text using spaCy NER and Matcher.
        
        Args:
            text: Raw text from OCR
            document_type: Type of document ('paystub' or 'bank_statement')
            
        Returns:
            Dictionary of extracted structured data
        """
        doc = self.nlp(text)
        extracted_data = {}
        
        # Use Matcher to find patterns
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            match_label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Extract the value (usually the last token which is the number)
            value_text = span.text
            
            if match_label == "NET_PAY":
                extracted_data["net_income"] = self._extract_currency(value_text)
            elif match_label == "GROSS_PAY":
                extracted_data["gross_income"] = self._extract_currency(value_text)
            elif match_label == "PAY_PERIOD":
                dates = self._extract_dates(value_text)
                if dates:
                    extracted_data["pay_period_start"] = dates[0]
                    if len(dates) > 1:
                        extracted_data["pay_period_end"] = dates[1]
            elif match_label == "ACCOUNT_BALANCE":
                extracted_data["account_balance"] = self._extract_currency(value_text)
        
        # Extract employer name using NER (if available)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                if "employer" not in extracted_data:
                    extracted_data["employer"] = ent.text
                    break
        
        # Extract bank name (usually at the top of statement)
        if document_type == "bank_statement":
            bank_names = ["Chase", "Bank of America", "Wells Fargo", "Citibank", 
                         "US Bank", "TD Bank", "Capital One"]
            for bank in bank_names:
                if bank.lower() in text.lower():
                    extracted_data["bank_name"] = bank
                    break
        
        return extracted_data
    
    def _extract_currency(self, text: str) -> Optional[float]:
        """Extract currency value from text."""
        # Remove non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d\.]', '', text)
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        # Find all date patterns
        date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'
        dates = re.findall(date_pattern, text)
        return dates
    
    def process_document(
        self,
        document_path: Union[str, Path],
        document_type: str = "paystub"
    ) -> Dict:
        """
        Complete pipeline: preprocess, OCR, and extract structured data.
        
        Args:
            document_path: Path to document file
            document_type: Type of document ('paystub' or 'bank_statement')
            
        Returns:
            Dictionary of extracted structured data
        """
        document_path = Path(document_path)
        
        # Determine file type and extract text
        if document_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(document_path)
        elif document_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            text = self.extract_text_from_image(document_path)
        else:
            raise ValueError(f"Unsupported file type: {document_path.suffix}")
        
        # Extract structured data
        structured_data = self.extract_structured_data(text, document_type)
        
        # Add raw text for downstream NLP processing
        structured_data['raw_text'] = text
        structured_data['document_type'] = document_type
        
        return structured_data


# Example usage
if __name__ == "__main__":
    # Initialize IDP engine
    idp = IDPEngine()
    
    # Process a pay stub
    result = idp.process_document("data/sample/paystub.pdf", document_type="paystub")
    print("Extracted Data:")
    print(result)
