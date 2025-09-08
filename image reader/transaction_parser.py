"""Transaction parsing module for bank statements."""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class TransactionParser:
    """Handles parsing of transactions from OCR text."""

    def __init__(self, config: Config):
        """
        Initialize the transaction parser.

        Args:
            config: Configuration object containing parsing settings
        """
        self.config = config

    def parse_single_line_transactions(self, text: str, statement_id: int) -> List[Dict]:
        """
        Parse transactions that appear on single lines using regex patterns.

        Args:
            text: OCR extracted text
            statement_id: ID of the statement being parsed

        Returns:
            List of parsed transaction dictionaries
        """
        transactions = []

        for pattern in self.config.transaction_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                if len(match) >= 4:
                    date1, date2, description, amount_str = match[:4]

                    transaction = self._create_transaction_dict(
                        statement_id, date1, amount_str, description.strip()
                    )
                    if transaction:
                        transactions.append(transaction)

        return transactions

    def parse_multiline_transactions(self, text: str, statement_id: int) -> List[Dict]:
        """
        Parse transactions that are split across multiple lines.

        Args:
            text: OCR extracted text
            statement_id: ID of the statement being parsed

        Returns:
            List of parsed transaction dictionaries
        """
        transactions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Look for transaction blocks between markers
        in_transaction_section = False
        transaction_lines = []

        for line in lines:
            if self.config.transaction_start_marker in line.upper():
                if in_transaction_section:
                    # Process accumulated transaction lines
                    transactions.extend(self._extract_transactions_from_block(
                        transaction_lines, statement_id))
                    transaction_lines = []
                in_transaction_section = True
            elif self.config.transaction_end_marker in line.upper():
                if in_transaction_section:
                    # Process final transaction block
                    transactions.extend(self._extract_transactions_from_block(
                        transaction_lines, statement_id))
                in_transaction_section = False
                break
            elif in_transaction_section:
                transaction_lines.append(line)

        return transactions

    def _extract_transactions_from_block(self, lines: List[str], statement_id: int) -> List[Dict]:
        """
        Extract transactions from a block of lines.

        Args:
            lines: List of text lines from a transaction block
            statement_id: ID of the statement being parsed

        Returns:
            List of parsed transaction dictionaries
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

                        transaction = self._create_transaction_dict(
                            statement_id, date1, amount, description)
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

            # Check if this line contains an amount at the end
            amount_match = re.search(r'(\d+[,\.]\d{1,2})-?\s*$', current_line)
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

    def _create_transaction_dict(self, statement_id: int, date_str: str,
                                amount_str: str, description: str) -> Optional[Dict]:
        """
        Create a transaction dictionary from parsed components.

        Args:
            statement_id: ID of the statement
            date_str: Date string in DD.MM format
            amount_str: Amount string
            description: Transaction description

        Returns:
            Transaction dictionary or None if parsing fails
        """
        try:
            # Parse amount
            amount_value = float(amount_str.replace(',', '.'))
            amount_value = -amount_value  # Make negative for expenses

            # Parse date
            formatted_date = self._parse_date(date_str)

            if not formatted_date:
                return None

            return {
                'statement_id': statement_id,
                'date': formatted_date,
                'amount': amount_value,
                'partner': self._identify_partner(description),
                'category': self._classify_category(description),
                'description': description
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
            return None

    def _identify_partner(self, description: str) -> str:
        """
        Identify transaction partner from description.

        Args:
            description: Transaction description

        Returns:
            Partner name or "Unknown"
        """
        description_upper = description.upper()

        for partner, pattern in self.config.partner_patterns.items():
            if re.search(pattern, description_upper):
                return partner

        return "Unknown"

    def _classify_category(self, description: str) -> str:
        """
        Classify transaction category based on description.

        Args:
            description: Transaction description

        Returns:
            Category name
        """
        description_upper = description.upper()

        for category, keywords in self.config.category_keywords.items():
            if any(keyword in description_upper for keyword in keywords):
                return category

        return 'Other'

    def parse_transactions(self, text: str, statement_id: int) -> List[Dict]:
        """
        Parse all transactions from OCR text using multiple strategies.

        Args:
            text: OCR extracted text
            statement_id: ID of the statement being parsed

        Returns:
            List of parsed transaction dictionaries
        """
        # First try single-line patterns
        transactions = self.parse_single_line_transactions(text, statement_id)

        # If no single-line transactions found, try multiline parsing
        if not transactions:
            transactions = self.parse_multiline_transactions(text, statement_id)

        # Remove duplicates based on date and amount
        unique_transactions = self._remove_duplicates(transactions)

        return unique_transactions

    def _remove_duplicates(self, transactions: List[Dict]) -> List[Dict]:
        """
        Remove duplicate transactions based on date and amount.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_transactions = []

        for trans in transactions:
            key = (trans['date'], trans['amount'])
            if key not in seen:
                seen.add(key)
                unique_transactions.append(trans)

        return unique_transactions
