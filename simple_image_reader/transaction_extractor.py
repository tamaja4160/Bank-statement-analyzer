"""Simple transaction extraction module - extracts raw transaction lines without categorization."""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

from .config import SimpleConfig

logger = logging.getLogger(__name__)

class SimpleTransactionExtractor:
    """Handles extraction of raw transaction lines from OCR text."""

    def __init__(self, config: SimpleConfig):
        """Initialize the transaction extractor."""
        self.config = config

    def extract_single_line_transactions(self, text: str) -> List[Dict]:
        """
        Extract transactions that appear on single lines using regex patterns.
        Skips any lines containing balance information (KONTOSTAND AM...).

        Args:
            text: OCR extracted text

        Returns:
            List of extracted transaction dictionaries with raw data
        """
        transactions = []
        processed_lines = set()  # Track which lines we've already processed

        # Split text into lines and filter out balance lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        filtered_lines = []

        for line in lines:
            # Skip balance lines
            if "KONTOSTAND AM" in line.upper():
                continue
            filtered_lines.append(line)

        # Reconstruct filtered text
        filtered_text = '\n'.join(filtered_lines)

        for pattern in self.config.transaction_patterns:
            matches = re.finditer(pattern, filtered_text, re.MULTILINE)
            for match in matches:
                # Get the full matched text to avoid duplicates
                full_match = match.group(0)
                if full_match in processed_lines:
                    continue

                processed_lines.add(full_match)

                if len(match.groups()) >= 4:
                    date1, date2, description, amount_str = match.groups()[:4]

                    transaction = self._create_simple_transaction(
                        date1, date2, amount_str, description.strip()
                    )
                    if transaction:
                        transactions.append(transaction)

        return transactions

    def extract_multiline_transactions(self, text: str) -> List[Dict]:
        """
        Extract transactions that are split across multiple lines.
        Skips the balance line (KONTOSTAND AM...) and starts parsing from actual transactions.

        Args:
            text: OCR extracted text

        Returns:
            List of extracted transaction dictionaries
        """
        transactions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Look for transaction blocks between markers
        in_transaction_section = False
        transaction_lines = []
        skip_next_line = False  # Flag to skip the balance line after finding KONTOSTAND

        for line in lines:
            if self.config.transaction_start_marker in line.upper():
                if in_transaction_section:
                    # Process accumulated transaction lines
                    transactions.extend(self._extract_from_block(transaction_lines))
                    transaction_lines = []
                in_transaction_section = True
                skip_next_line = True  # Skip the next line (balance line)
            elif self.config.transaction_end_marker in line.upper():
                if in_transaction_section:
                    # Process final transaction block
                    transactions.extend(self._extract_from_block(transaction_lines))
                in_transaction_section = False
                break
            elif in_transaction_section:
                if skip_next_line:
                    # Skip the balance line (KONTOSTAND AM...)
                    skip_next_line = False
                    continue
                elif "KONTOSTAND AM" in line.upper():
                    # Additional check to skip any balance lines
                    continue
                else:
                    # This is a transaction line
                    transaction_lines.append(line)

        return transactions

    def _extract_from_block(self, lines: List[str]) -> List[Dict]:
        """
        Extract transactions from a block of lines.
        
        Args:
            lines: List of text lines from a transaction block
            
        Returns:
            List of extracted transaction dictionaries
        """
        transactions = []
        i = 0

        while i < len(lines):
            # Look for date pattern
            date_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?\s*$', lines[i])
            if date_match and i + 1 < len(lines):
                date1 = date_match.group(1)

                # Check next line for second date
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                date2_match = re.match(r'^(\d{1,2}\.\d{1,2})\.?\s*$', next_line)

                if date2_match and i + 2 < len(lines):
                    date2 = date2_match.group(1)

                    # Look for description and amount in subsequent lines
                    description_parts, amount = self._extract_description_and_amount(
                        lines, i + 2)

                    if amount and description_parts:
                        description = ' '.join(description_parts)

                        transaction = self._create_simple_transaction(
                            date1, date2, amount, description)
                        if transaction:
                            transactions.append(transaction)

                        # Skip processed lines
                        i = max(i + 2, i + len(description_parts) + 1)
                        continue

            i += 1

        return transactions

    def _extract_description_and_amount(self, lines: List[str], start_idx: int) -> Tuple[List[str], Optional[str]]:
        """
        Extract description parts and amount from lines.
        
        Args:
            lines: List of text lines
            start_idx: Starting index to search from
            
        Returns:
            Tuple of (description_parts, amount_string)
        """
        description_parts = []
        amount = None
        max_lookahead = min(5, len(lines) - start_idx)  # Look ahead max 5 lines

        for j in range(max_lookahead):
            current_idx = start_idx + j
            if current_idx >= len(lines):
                break

            current_line = lines[current_idx]

            # Check if this line contains an amount at the end (flexible ending)
            amount_match = re.search(r'(\d+[,\.]\d{1,2})[-\.\s]*$', current_line)
            if amount_match:
                amount_str = amount_match.group(1)
                # Remove amount from description
                desc_part = re.sub(r'\s*\d+[,\.]\d{1,2}-?\s*$', '', current_line).strip()
                if desc_part:
                    description_parts.append(desc_part)
                amount = amount_str
                break
            else:
                # This line is part of description
                if current_line and not re.match(r'^\d{1,2}\.\d{1,2}\.?\s*$', current_line):
                    description_parts.append(current_line)

        return description_parts, amount

    def _create_simple_transaction(self, date1: str, date2: str, amount_str: str, description: str) -> Optional[Dict]:
        """
        Create a simple transaction dictionary from parsed components.
        
        Args:
            date1: First date string in DD.MM format
            date2: Second date string in DD.MM format  
            amount_str: Amount string
            description: Transaction description
            
        Returns:
            Simple transaction dictionary or None if parsing fails
        """
        try:
            # Parse amount (keep as string and float)
            amount_value = float(amount_str.replace(',', '.'))
            
            # Parse date (keep simple)
            formatted_date = self._parse_date(date1)
            
            if not formatted_date:
                return None

            return {
                'date1': date1,
                'date2': date2,
                'formatted_date': formatted_date,
                'amount_str': amount_str,
                'amount_value': amount_value,
                'description': description,
                'raw_line': f"{date1}. {date2}. {description} {amount_str}-"
            }

        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse transaction: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string into standardized format.
        
        Args:
            date_str: Date string in DD.MM format
            
        Returns:
            Formatted date string or None if parsing fails
        """
        try:
            current_year = datetime.now().year
            date_obj = datetime.strptime(f"{date_str}.{current_year}", "%d.%m.%Y")
            return date_obj.strftime("%d.%m.%Y")
        except ValueError:
            logger.warning(f"Failed to parse date: {date_str}")
            return f"{date_str}.{datetime.now().year}"  # Fallback

    def extract_transactions(self, text: str) -> List[Dict]:
        """
        Extract all transactions from OCR text using multiple strategies.
        
        Args:
            text: OCR extracted text
            
        Returns:
            List of extracted transaction dictionaries
        """
        # First try single-line patterns
        transactions = self.extract_single_line_transactions(text)

        # If no single-line transactions found, try multiline parsing
        if not transactions:
            transactions = self.extract_multiline_transactions(text)

        # Remove duplicates based on date and amount
        unique_transactions = self._remove_duplicates(transactions)

        return unique_transactions

    def _remove_duplicates(self, transactions: List[Dict]) -> List[Dict]:
        """
        Remove duplicate transactions based on description and amount.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_transactions = []

        for trans in transactions:
            # Use description and amount for duplicate detection (more robust than date)
            key = (trans['description'].strip(), trans['amount_value'])
            if key not in seen:
                seen.add(key)
                unique_transactions.append(trans)

        return unique_transactions
