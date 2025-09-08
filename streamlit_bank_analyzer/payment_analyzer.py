"""
Payment analyzer for detecting recurring payments and recommending IBAN changes.
Uses fuzzy matching and clustering to identify similar transactions.
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import statistics
from difflib import SequenceMatcher
import pandas as pd

def analyze_recurring_payments(transactions: List[Dict]) -> Dict:
    """
    Analyze transactions to identify recurring payments.

    Args:
        transactions: List of transaction dictionaries from OCR extraction

    Returns:
        Dictionary containing analysis results and recommendations
    """
    if not transactions:
        return {'recurring_payments': [], 'recommendations': []}

    # Group transactions by similar descriptions
    grouped_transactions = group_similar_transactions(transactions)

    # Filter for recurring patterns
    recurring_payments = []
    for description_group, group_transactions in grouped_transactions.items():
        if len(group_transactions) >= 2:  # At least 2 occurrences
            analysis = analyze_payment_pattern(group_transactions)
            if analysis['is_recurring']:
                recurring_payments.append({
                    'description': description_group,
                    'occurrences': len(group_transactions),
                    'average_amount': analysis['average_amount'],
                    'amount_range': analysis['amount_range'],
                    'confidence': analysis['confidence'],
                    'transactions': group_transactions,
                    'pattern_type': analysis['pattern_type']
                })

    # Sort by confidence and occurrences
    recurring_payments.sort(key=lambda x: (x['confidence'], x['occurrences']), reverse=True)

    # Generate recommendations
    recommendations = generate_iban_recommendations(recurring_payments)

    return {
        'recurring_payments': recurring_payments,
        'recommendations': recommendations,
        'total_analyzed': len(transactions),
        'unique_descriptions': len(grouped_transactions)
    }

def group_similar_transactions(transactions: List[Dict], similarity_threshold: float = 0.8) -> Dict[str, List[Dict]]:
    """
    Group transactions by similar descriptions using fuzzy matching.

    Args:
        transactions: List of transaction dictionaries
        similarity_threshold: Minimum similarity score (0-1) for grouping

    Returns:
        Dictionary mapping canonical descriptions to lists of transactions
    """
    groups = defaultdict(list)

    # Sort transactions by description length (longer first for better matching)
    sorted_transactions = sorted(transactions, key=lambda x: len(x.get('description', '')), reverse=True)

    for transaction in sorted_transactions:
        description = transaction.get('description', '').strip()
        if not description:
            continue

        # Find best matching existing group
        best_match = None
        best_score = 0

        for canonical_desc in groups.keys():
            score = calculate_description_similarity(description, canonical_desc)
            if score > best_score and score >= similarity_threshold:
                best_match = canonical_desc
                best_score = score

        if best_match:
            groups[best_match].append(transaction)
        else:
            # Create new group with cleaned description as canonical
            canonical = clean_description_for_grouping(description)
            groups[canonical].append(transaction)

    return dict(groups)

def calculate_description_similarity(desc1: str, desc2: str) -> float:
    """
    Calculate similarity between two transaction descriptions.

    Args:
        desc1, desc2: Transaction descriptions to compare

    Returns:
        Similarity score between 0 and 1
    """
    if not desc1 or not desc2:
        return 0.0

    # Clean descriptions for comparison
    clean1 = clean_description_for_comparison(desc1)
    clean2 = clean_description_for_comparison(desc2)

    # Use sequence matcher for fuzzy matching
    matcher = SequenceMatcher(None, clean1, clean2)
    return matcher.ratio()

def clean_description_for_comparison(description: str) -> str:
    """
    Clean description for similarity comparison by removing variable parts.

    Args:
        description: Raw transaction description

    Returns:
        Cleaned description for comparison
    """
    # Convert to uppercase for consistent matching
    desc = description.upper()

    # Remove common variable patterns
    patterns_to_remove = [
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # Dates
        r'\d{8,}',  # Contract numbers
        r'[A-Z]{6}',  # Random strings
        r'\d{4,}',  # Long numbers
        r'K-\d+',  # Contract suffixes
    ]

    for pattern in patterns_to_remove:
        desc = re.sub(pattern, '', desc)

    # Remove extra whitespace and common separators
    desc = re.sub(r'\s+', ' ', desc).strip()
    desc = re.sub(r'[/-]', ' ', desc)

    return desc

def clean_description_for_grouping(description: str) -> str:
    """
    Clean description to create a canonical form for grouping.

    Args:
        description: Raw transaction description

    Returns:
        Canonical description for the group
    """
    # Extract key merchant/partner name
    desc = description.upper()

    # Look for known patterns
    merchant_patterns = [
        r'(VODAFONE GMBH|ALLIANZ SE|ZEUS BODYPOWER|STADTWERKE.*)',
        r'(EDEKA|REWE|LIDL|ALDI|BURGER KING|MCDONALDS)',
        r'(SHELL|ARAL|JET)',
        r'(AMAZON|ZALANDO|EBAY)',
        r'(PAYPAL)',
    ]

    for pattern in merchant_patterns:
        match = re.search(pattern, desc)
        if match:
            return match.group(1).strip()

    # Fallback: take first significant word or phrase
    words = re.findall(r'\b[A-Z]{3,}\b', desc)
    if words:
        return words[0]

    # Last resort: first part of description
    return desc.split()[0] if desc.split() else desc

def analyze_payment_pattern(transactions: List[Dict]) -> Dict:
    """
    Analyze a group of transactions to determine if they represent recurring payments.

    Args:
        transactions: List of transactions in the same group

    Returns:
        Analysis results dictionary
    """
    if len(transactions) < 2:
        return {'is_recurring': False, 'confidence': 0.0}

    # Extract amounts
    amounts = []
    for tx in transactions:
        amount_str = tx.get('amount_str', '0')
        try:
            amount = float(amount_str.replace(',', '.'))
            amounts.append(abs(amount))  # Use absolute value
        except (ValueError, AttributeError):
            continue

    if len(amounts) < 2:
        return {'is_recurring': False, 'confidence': 0.0}

    # Calculate amount statistics
    avg_amount = statistics.mean(amounts)
    amount_variance = statistics.variance(amounts) if len(amounts) > 1 else 0
    amount_range = max(amounts) - min(amounts)

    # Determine if amounts are similar (within 5% tolerance)
    amount_similarity = amount_range / avg_amount if avg_amount > 0 else 1.0
    amounts_similar = amount_similarity <= 0.10  # 5% tolerance

    # Calculate confidence based on various factors
    confidence = 0.0

    # Factor 1: Number of occurrences (more = higher confidence)
    occurrence_factor = min(len(transactions) / 5.0, 1.0)  # Cap at 5 occurrences

    # Factor 2: Amount consistency
    consistency_factor = 1.0 - min(amount_similarity, 1.0)

    # Factor 3: Known recurring patterns in description
    description = transactions[0].get('description', '').upper()
    pattern_factor = 0.0
    if any(keyword in description for keyword in ['BEITRAG', 'RECHNUNG', 'ABSCHLAG', 'MITGLIED']):
        pattern_factor = 0.8
    elif any(keyword in description for keyword in ['KARTENZAHLUNG', 'MARTKPLC']):
        pattern_factor = 0.6

    # Combine factors
    confidence = (occurrence_factor * 0.4 + consistency_factor * 0.4 + pattern_factor * 0.2)

    # Determine pattern type
    if pattern_factor >= 0.8:
        pattern_type = "Subscription/Service"
    elif amounts_similar and len(transactions) >= 3:
        pattern_type = "Regular Recurring"
    elif len(transactions) >= 2:
        pattern_type = "Potential Recurring"
    else:
        pattern_type = "Irregular"

    return {
        'is_recurring': confidence >= 0.6 and amounts_similar,
        'confidence': confidence,
        'average_amount': avg_amount,
        'amount_range': amount_range,
        'amount_similarity': amount_similarity,
        'pattern_type': pattern_type,
        'amounts': amounts
    }

def generate_iban_recommendations(recurring_payments: List[Dict]) -> List[str]:
    """
    Generate IBAN change recommendations based on recurring payments.

    Args:
        recurring_payments: List of recurring payment analyses

    Returns:
        List of recommendation strings
    """
    if not recurring_payments:
        return ["No recurring payments identified that would benefit from IBAN changes."]

    recommendations = []

    # Group by confidence level
    high_confidence = [p for p in recurring_payments if p['confidence'] >= 0.8]
    medium_confidence = [p for p in recurring_payments if 0.6 <= p['confidence'] < 0.8]

    if high_confidence:
        billers = [p['description'] for p in high_confidence]
        recommendations.append(f"💡 **High Priority**: We will change your IBAN for these regular billers: {', '.join(billers)}")

    if medium_confidence:
        billers = [p['description'] for p in medium_confidence]
        recommendations.append(f"🔄 **Medium Priority**: Consider IBAN changes for: {', '.join(billers)}")

    # Add summary statistics
    total_recurring = len(recurring_payments)
    total_occurrences = sum(p['occurrences'] for p in recurring_payments)
    avg_monthly = sum(p['average_amount'] for p in recurring_payments)

    recommendations.append("\n📊 **Summary**:")
    recommendations.append(f"   • {total_recurring} recurring payment types identified")
    recommendations.append(f"   • {total_occurrences} total recurring transactions")
    recommendations.append(f"   • Average monthly recurring amount: €{avg_monthly:.2f}")

    return recommendations
