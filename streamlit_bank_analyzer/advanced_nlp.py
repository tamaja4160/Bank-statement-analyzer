"""
Advanced NLP module using Transformers for transaction analysis.
Implements BERT-based semantic similarity and Named Entity Recognition.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

class AdvancedNLPProcessor:
    """
    Advanced NLP processor using transformers for transaction analysis.
    """

    def __init__(self):
        """Initialize the NLP models."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize Sentence-BERT for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"Warning: Could not load Sentence-BERT model: {e}")
            self.sentence_model = None

        # Initialize German NER model
        try:
            self.ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-german")
            self.ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-german")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.ner_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            self.ner_pipeline = None

    def get_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get semantic embeddings for transaction descriptions.

        Args:
            texts: List of transaction descriptions

        Returns:
            Numpy array of embeddings
        """
        if self.sentence_model is None:
            # Fallback to simple TF-IDF style features
            return self._fallback_embeddings(texts)

        try:
            embeddings = self.sentence_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return self._fallback_embeddings(texts)

    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Fallback embedding generation when Sentence-BERT fails.

        Args:
            texts: List of transaction descriptions

        Returns:
            Simple bag-of-words style embeddings
        """
        # Create simple features based on text characteristics
        features = []
        for text in texts:
            text = text.upper()
            feature_vector = [
                len(text),  # Length
                text.count(' '),  # Word count
                1 if 'STROM' in text else 0,  # Electricity indicator
                1 if 'GAS' in text else 0,    # Gas indicator
                1 if 'ABSCHLAG' in text else 0,  # Installment indicator
                1 if 'RECHNUNG' in text else 0,  # Bill indicator
                len(re.findall(r'\d{6,12}', text)),  # Contract number count
            ]
            features.append(feature_vector)

        return np.array(features)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from transaction description.

        Args:
            text: Transaction description

        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'MERCHANT': [],
            'CONTRACT_ID': [],
            'AMOUNT': [],
            'PAYMENT_TYPE': []
        }

        if self.ner_pipeline is None:
            # Fallback entity extraction
            return self._fallback_entity_extraction(text)

        try:
            # Run NER
            ner_results = self.ner_pipeline(text)

            # Process NER results
            current_entity = []
            current_label = None

            for result in ner_results:
                label = result['entity'].split('-')[-1]  # Remove B-/I- prefix
                word = result['word']

                if label in ['ORG', 'MISC']:  # Organizations and miscellaneous entities
                    if current_entity and current_label == label:
                        current_entity.append(word)
                    else:
                        if current_entity:
                            entities['MERCHANT'].append(' '.join(current_entity))
                        current_entity = [word]
                        current_label = label
                else:
                    if current_entity:
                        entities['MERCHANT'].append(' '.join(current_entity))
                        current_entity = []
                        current_label = None

            if current_entity:
                entities['MERCHANT'].append(' '.join(current_entity))

        except Exception as e:
            print(f"Error in NER: {e}")
            return self._fallback_entity_extraction(text)

        # Extract contract numbers and payment types
        entities.update(self._fallback_entity_extraction(text))

        return entities

    def _fallback_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """
        Fallback entity extraction using regex patterns.

        Args:
            text: Transaction description

        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'MERCHANT': [],
            'CONTRACT_ID': [],
            'AMOUNT': [],
            'PAYMENT_TYPE': []
        }

        # Extract contract numbers
        contract_matches = re.findall(r'\d{6,12}', text)
        entities['CONTRACT_ID'] = contract_matches

        # Extract amounts
        amount_matches = re.findall(r'\d+[,.]\d{2}', text)
        entities['AMOUNT'] = amount_matches

        # Identify payment types
        text_upper = text.upper()
        if 'ABSCHLAG' in text_upper:
            entities['PAYMENT_TYPE'].append('INSTALLMENT')
        if 'STROM' in text_upper:
            entities['PAYMENT_TYPE'].append('ELECTRICITY')
        if 'GAS' in text_upper:
            entities['PAYMENT_TYPE'].append('GAS')
        if 'RECHNUNG' in text_upper:
            entities['PAYMENT_TYPE'].append('BILL')

        # Extract merchant names (simplified)
        # Look for common utility company patterns
        merchant_patterns = [
            r'ABSCHLAG\s+(.*?)(?:\s+\d|$)',
            r'(?:STADTWERKE|SW)\s+(\w+)',
            r'([A-Z][A-Z\s]+?)(?:\s+\d|$)'
        ]

        for pattern in merchant_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                if len(match.strip()) > 3:  # Avoid very short matches
                    entities['MERCHANT'].append(match.strip())

        return entities

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1, text2: Texts to compare

        Returns:
            Similarity score between 0 and 1
        """
        if self.sentence_model is None:
            # Fallback to simple string similarity
            return self._calculate_string_similarity(text1, text2)

        try:
            embeddings = self.sentence_model.encode([text1, text2], convert_to_numpy=True)
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return self._calculate_string_similarity(text1, text2)

    def _calculate_string_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple string similarity as fallback.

        Args:
            text1, text2: Texts to compare

        Returns:
            Similarity score between 0 and 1
        """
        # Clean texts
        clean1 = self._clean_text_for_similarity(text1)
        clean2 = self._clean_text_for_similarity(text2)

        # Use sequence matcher
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, clean1, clean2)
        return matcher.ratio()

    def _clean_text_for_similarity(self, text: str) -> str:
        """
        Clean text for similarity comparison.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Convert to uppercase
        text = text.upper()

        # Remove contract numbers and variable parts
        patterns_to_remove = [
            r'\d{6,12}',  # Contract numbers
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # Dates
            r'K-\d+',  # Contract suffixes
            r'/\d+',  # Reference numbers
            r'-\d+',  # Reference numbers
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def classify_transaction_type(self, text: str) -> Dict[str, float]:
        """
        Classify transaction type using semantic analysis.

        Args:
            text: Transaction description

        Returns:
            Dictionary of transaction types and confidence scores
        """
        text_upper = text.upper()

        # Define transaction type keywords
        type_keywords = {
            'UTILITY_ELECTRICITY': ['STROM', 'ELEKTRIZITÄT', 'ELECTRICITY'],
            'UTILITY_GAS': ['GAS', 'ERDGAS'],
            'INSURANCE': ['VERSICHERUNG', 'INSURANCE'],
            'SUBSCRIPTION': ['ABO', 'SUBSCRIPTION', 'MITGLIEDSBEITRAG'],
            'GROCERIES': ['REWE', 'ALDI', 'LIDL', 'EDeka'],
            'TRANSPORT': ['DB', 'BAHN', 'TICKET'],
            'ENTERTAINMENT': ['NETFLIX', 'AMAZON', 'SPOTIFY']
        }

        scores = {}
        for tx_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_upper)
            if score > 0:
                scores[tx_type] = min(score / len(keywords), 1.0)  # Normalize

        # Special handling for ABSCHLAG payments
        if 'ABSCHLAG' in text_upper:
            if 'STROM' in text_upper:
                scores['UTILITY_ELECTRICITY'] = 1.0
            elif 'GAS' in text_upper:
                scores['UTILITY_GAS'] = 1.0
            else:
                scores['UTILITY_GENERAL'] = 1.0

        return scores
