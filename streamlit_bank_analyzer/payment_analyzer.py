"""
Simple and efficient payment analyzer for detecting recurring payments.
"""

import re
from typing import List, Dict
from collections import defaultdict

def analyze_recurring_payments(transactions: List[Dict]) -> Dict:
    """Simple analysis to identify recurring payments."""
    if not transactions:
        return {'recurring_payments': [], 'recommendations': []}

    # Group by cleaned description
    groups = defaultdict(list)
    for tx in transactions:
        desc = clean_description(tx.get('description', ''))
        groups[desc].append(tx)

    # Find recurring payments (simple logic with 10% tolerance)
    recurring_payments = []
    for description, group_txs in groups.items():
        if len(group_txs) >= 2:  # At least 2 occurrences
            avg_amount = get_average_amount(group_txs)
            amount_range = get_amount_range(group_txs)

            # Check if amount variation is within 10% tolerance
            if avg_amount > 0:
                variation_percentage = (amount_range / avg_amount) * 100
                if variation_percentage <= 10.0:  # Maximum 10% tolerance
                    recurring_payments.append({
                        'description': description,
                        'occurrences': len(group_txs),
                        'average_amount': avg_amount,
                        'transactions': group_txs,
                        'confidence': 0.85,  # Simple confidence score
                        'pattern_type': get_pattern_type(description),
                        'amount_range': amount_range
                    })

    # Sort by occurrences
    recurring_payments.sort(key=lambda x: x['occurrences'], reverse=True)

    # Import deal comparison here to avoid circular imports
    find_better_deals = None
    try:
        # Try absolute import when modules are imported as top-level files
        from deal_comparison import find_better_deals as _fbd
        find_better_deals = _fbd
    except Exception:
        try:
            # Fallback to relative import when used as a package
            from .deal_comparison import find_better_deals as _fbd
            find_better_deals = _fbd
        except Exception:
            find_better_deals = None

    if find_better_deals:
        deal_analysis = find_better_deals(recurring_payments)
    else:
        deal_analysis = {
            'deal_comparisons': [],
            'total_current_cost': 0,
            'total_monthly_savings': 0,
            'total_annual_savings': 0,
            'savings_percentage': 0
        }

    recommendations = []
    if recurring_payments:
        recommendations.append(f"Detected {len(recurring_payments)} recurring payment(s).")
        if deal_analysis['total_monthly_savings'] > 0:
            recommendations.append(f"💰 Potential monthly savings: €{deal_analysis['total_monthly_savings']:.2f}")
    else:
        recommendations.append("No recurring payments detected.")

    return {
        'recurring_payments': recurring_payments,
        'deal_analysis': deal_analysis,
        'recommendations': recommendations,
        'total_analyzed': len(transactions),
        'unique_descriptions': len(groups)
    }

def clean_description(description: str) -> str:
    """Clean transaction description for grouping."""
    if not description:
        return ""

    desc = description.upper()

    # Remove contract numbers and dates
    desc = re.sub(r'\d{6,12}', '', desc)  # Contract numbers
    desc = re.sub(r'\d{1,2}\.\d{1,2}\.\d{2,4}', '', desc)  # Dates
    desc = re.sub(r'K-\d+', '', desc)  # Contract suffixes

    # Clean up whitespace
    desc = re.sub(r'\s+', ' ', desc).strip()

    return desc

def get_average_amount(transactions: List[Dict]) -> float:
    """Calculate average amount from transactions."""
    amounts = []
    for tx in transactions:
        amount_str = tx.get('amount_str', '0')
        try:
            amount = float(amount_str.replace(',', '.'))
            amounts.append(abs(amount))
        except (ValueError, AttributeError):
            continue

    return sum(amounts) / len(amounts) if amounts else 0.0

def get_amount_range(transactions: List[Dict]) -> float:
    """Calculate amount range (max - min) from transactions."""
    amounts = []
    for tx in transactions:
        amount_str = tx.get('amount_str', '0')
        try:
            amount = float(amount_str.replace(',', '.'))
            amounts.append(abs(amount))
        except (ValueError, AttributeError):
            continue

    if len(amounts) < 2:
        return 0.0

    return max(amounts) - min(amounts)

# Pattern type constants
PATTERN_TYPE_SUBSCRIPTION = 'Subscription/Service'
PATTERN_TYPE_REGULAR = 'Regular Recurring'

def get_pattern_type(description: str) -> str:
    """Determine the pattern type based on description."""
    desc_upper = description.upper()
    if 'BEITRAG' in desc_upper or 'ABO' in desc_upper:
        return PATTERN_TYPE_SUBSCRIPTION
    return PATTERN_TYPE_REGULAR
