"""
Deal comparison module for finding cheaper alternatives to recurring payments.
Provides realistic price comparisons based on German market data.
"""

from typing import Dict, List, Optional
import random

# Alternative providers database with realistic German market prices
ALTERNATIVE_PROVIDERS = {
    # Insurance alternatives
    "Liability Insurance": [
        {"provider": "HUK24", "price": 35.50, "savings": 20.0, "rating": 4.2},
        {"provider": "DEVK", "price": 42.80, "savings": 12.7, "rating": 4.0},
        {"provider": "CosmosDirekt", "price": 38.90, "savings": 16.6, "rating": 3.8},
    ],
    "Car Insurance": [
        {"provider": "HUK-Coburg", "price": 75.30, "savings": 14.2, "rating": 4.1},
        {"provider": "CosmosDirekt", "price": 69.90, "savings": 19.6, "rating": 3.9},
        {"provider": "DEVK", "price": 82.40, "savings": 7.1, "rating": 4.0},
    ],
    "Home Insurance": [
        {"provider": "HUK24", "price": 32.50, "savings": 13.3, "rating": 4.0},
        {"provider": "CosmosDirekt", "price": 35.80, "savings": 10.0, "rating": 3.7},
        {"provider": "DEVK", "price": 38.20, "savings": 7.6, "rating": 3.9},
    ],
    "Legal Protection": [
        {"provider": "DEVK", "price": 25.90, "savings": 3.1, "rating": 4.1},
        {"provider": "ARAG", "price": 22.50, "savings": 6.5, "rating": 4.3},
        {"provider": "HUK-Coburg", "price": 24.80, "savings": 4.2, "rating": 3.8},
    ],
    "Health Insurance": [
        {"provider": "TK", "price": 75.40, "savings": 14.1, "rating": 3.9},
        {"provider": "Barmer", "price": 72.80, "savings": 16.7, "rating": 3.7},
        {"provider": "DAK Gesundheit", "price": 78.90, "savings": 10.6, "rating": 4.0},
    ],

    # Telecom alternatives
    "Internet": [
        {"provider": "Vodafone", "price": 39.99, "savings": 9.96, "rating": 3.8},
        {"provider": "1&1", "price": 34.99, "savings": 14.96, "rating": 3.9},
        {"provider": "O2 Germany", "price": 29.99, "savings": 19.96, "rating": 3.6},
    ],
    "Mobile": [
        {"provider": "Aldi Talk", "price": 7.99, "savings": 16.01, "rating": 4.0},
        {"provider": "Vodafone", "price": 19.99, "savings": 4.01, "rating": 3.8},
        {"provider": "O2 Germany", "price": 14.99, "savings": 9.01, "rating": 3.7},
    ],

    # Utilities alternatives
    "Electricity": [
        {"provider": "Tibber", "price": 25.90, "savings": 52.5, "rating": 4.2},
        {"provider": "Octopus Energy", "price": 29.50, "savings": 48.9, "rating": 4.1},
        {"provider": "Grünwelt Energie", "price": 32.80, "savings": 46.1, "rating": 3.9},
        {"provider": "E.ON Energie", "price": 78.40, "savings": 0.0, "rating": 3.5},  # Current provider (no savings)
    ],
    "Gas": [
        {"provider": "E.ON Energie", "price": 58.90, "savings": 10.5, "rating": 3.5},
        {"provider": "Vattenfall", "price": 62.40, "savings": 5.0, "rating": 3.7},
        {"provider": "Stadtwerke Berlin", "price": 65.80, "savings": 0.0, "rating": 3.6},  # Current provider (no savings)
    ],
    "Water": [
        {"provider": "Eckwater", "price": 22.50, "savings": 13.1, "rating": 4.0},
        {"provider": "Wasserzweckverband", "price": 24.80, "savings": 4.2, "rating": 3.9},
        {"provider": "Stadtwerke Hamburg", "price": 25.90, "savings": 0.0, "rating": 3.8},  # Current provider (no savings)
    ],

    # Streaming alternatives
    "Entertainment": [
        {"provider": "Disney+", "price": 8.99, "savings": 0.0, "rating": 4.3},
        {"provider": "MagentaTV", "price": 10.00, "savings": -1.01, "rating": 3.8},
        {"provider": "Joyn", "price": 6.99, "savings": 2.0, "rating": 3.9},
    ],
    "Music": [
        {"provider": "Spotify", "price": 9.99, "savings": 0.0, "rating": 4.4},
        {"provider": "Amazon Music", "price": 7.99, "savings": 2.0, "rating": 4.1},
        {"provider": "Apple Music", "price": 9.99, "savings": 0.0, "rating": 4.2},
    ],

    # Software alternatives
    "Software": [
        {"provider": "Microsoft", "price": 69.00, "savings": 0.0, "rating": 4.1},
        {"provider": "Google Workspace", "price": 12.00, "savings": 57.0, "rating": 4.3},
        {"provider": "Zoho Workplace", "price": 18.00, "savings": 51.0, "rating": 4.0},
    ],
}

def find_better_deals(recurring_payments: List[Dict]) -> Dict:
    """
    Find better deals for recurring payments.

    Args:
        recurring_payments: List of recurring payment dictionaries

    Returns:
        Dictionary with deal comparisons and savings analysis
    """
    deal_comparisons = []
    total_current_cost = 0
    total_potential_savings = 0

    print(f"DEBUG: Analyzing {len(recurring_payments)} recurring payments")

    for payment in recurring_payments:
        current_amount = payment['average_amount']
        total_current_cost += current_amount

        description = payment['description']
        print(f"DEBUG: Checking payment: {description}")

        # Try to match the payment to a category
        category = _match_payment_to_category(description)
        print(f"DEBUG: Matched category: {category}")

        if category and category in ALTERNATIVE_PROVIDERS:
            alternatives = ALTERNATIVE_PROVIDERS[category]
            print(f"DEBUG: Found {len(alternatives)} alternatives for {category}")

            # Find the best alternative (highest savings)
            best_alternative = max(alternatives, key=lambda x: x['savings'])
            print(f"DEBUG: Best alternative: {best_alternative['provider']} with {best_alternative['savings']}% savings")

            # Calculate actual savings for this payment based on price difference (not percentage heuristic)
            alternative_price = best_alternative['price']
            monthly_savings = max(0.0, current_amount - alternative_price)
            if monthly_savings > 0:
                annual_savings = monthly_savings * 12

                deal_comparisons.append({
                    'current_provider': description.split()[0],  # First word
                    'current_amount': round(current_amount, 2),
                    'category': category,
                    'alternative_provider': best_alternative['provider'],
                    'alternative_amount': round(alternative_price, 2),
                    'monthly_savings': round(monthly_savings, 2),
                    'annual_savings': round(annual_savings, 2),
                    'savings_percentage': round((monthly_savings / current_amount) * 100, 1),
                    'rating': best_alternative['rating']
                })

                total_potential_savings += monthly_savings
                print(f"DEBUG: Added deal comparison with €{monthly_savings:.2f} monthly savings")

    # Sort by potential savings (highest first)
    deal_comparisons.sort(key=lambda x: x['monthly_savings'], reverse=True)

    print(f"DEBUG: Found {len(deal_comparisons)} deal comparisons")

    return {
        'deal_comparisons': deal_comparisons,
        'total_current_cost': round(total_current_cost, 2),
        'total_monthly_savings': round(total_potential_savings, 2),
        'total_annual_savings': round(total_potential_savings * 12, 2),
        'savings_percentage': round((total_potential_savings / total_current_cost) * 100, 1) if total_current_cost > 0 else 0
    }

def _match_payment_to_category(description: str) -> Optional[str]:
    """
    Match a payment description to a category for deal comparison.

    Args:
        description: Payment description

    Returns:
        Category name or None if no match
    """
    desc_lower = description.lower()

    # Insurance mappings - match the actual generated descriptions
    if any(word in desc_lower for word in ['haftpflicht', 'haftpflichtversicherung', 'allianz se', 'allianz']):
        return "Liability Insurance"
    if any(word in desc_lower for word in ['kfz', 'kaskoversicherung', 'autoversicherung', 'huk-coburg', 'huk', 'coburg']):
        return "Car Insurance"
    if any(word in desc_lower for word in ['wohngebaeude', 'hausversicherung', 'wohnungsversicherung', 'generali']):
        return "Home Insurance"
    if any(word in desc_lower for word in ['rechtsschutz', 'rechtsschutzversicherung', 'axa']):
        return "Legal Protection"
    if any(word in desc_lower for word in ['krankenversicherung', 'gesundheitsversicherung', 'dkv deutsche krank', 'dkv']):
        return "Health Insurance"

    # Telecom mappings - match the actual generated descriptions
    if any(word in desc_lower for word in ['telekom', 'vodafone', 'internet', 'dsl', 'deutsche telekom', 'telekom rechnung']):
        return "Internet"
    if any(word in desc_lower for word in ['mobilfunk', 'mobile', 'handy', 'o2 germany', 'o2', '1&1', '1und1']):
        return "Mobile"

    # Utilities mappings - match the actual generated descriptions
    if any(word in desc_lower for word in ['strom', 'electricity', 'e.on', 'rwe', 'enbw', 'e.on energie', 'energie']):
        return "Electricity"
    if any(word in desc_lower for word in ['wasser', 'water', 'wasserabschlag', 'wasserrechnung']):
        return "Water"
    if any(word in desc_lower for word in ['gas', 'gasabschlag', 'gasrechnung']):
        return "Gas"
    if any(word in desc_lower for word in ['stadtwerke', 'stadtwerke berlin', 'stadtwerke muenchen', 'stadtwerke hamburg', 'stadtwerke koeln']):
        # Additional logic for Stadtwerke - check context
        if 'wasser' in desc_lower or 'water' in desc_lower:
            return "Water"
        elif 'gas' in desc_lower:
            return "Gas"
        else:
            return "Electricity"  # Default for Stadtwerke

    # Entertainment mappings - match the actual generated descriptions
    if any(word in desc_lower for word in ['netflix', 'amazon', 'disney', 'entertainment', 'amazon prime', 'disney plus', 'dazn', 'sport']):
        return "Entertainment"
    if any(word in desc_lower for word in ['spotify', 'music', 'apple music']):
        return "Music"

    # Software mappings - match the actual generated descriptions
    if any(word in desc_lower for word in ['microsoft', 'adobe', 'software', 'microsoft 365', 'creative cloud']):
        return "Software"

    # Additional mappings for other services
    if any(word in desc_lower for word in ['dropbox', 'google one', 'cloud']):
        return "Software"  # Map cloud services to Software category
    if any(word in desc_lower for word in ['hellofresh', 'food delivery']):
        return "Entertainment"  # Map food delivery to Entertainment
    if any(word in desc_lower for word in ['bookbeat', 'audible', 'books', 'audiobooks']):
        return "Entertainment"  # Map books/audiobooks to Entertainment
    if any(word in desc_lower for word in ['ikea family', 'shopping']):
        return "Entertainment"  # Map shopping to Entertainment
    if any(word in desc_lower for word in ['zeus bodypower', 'gym']):
        return "Entertainment"  # Map gym to Entertainment
    if any(word in desc_lower for word in ['ing dibba', 'banking']):
        return "Entertainment"  # Map banking to Entertainment
    if any(word in desc_lower for word in ['deutsche bahn', 'transportation']):
        return "Entertainment"  # Map transportation to Entertainment

    # Fallback: Try to match based on common keywords
    fallback_category = _fallback_category_match(description)
    if fallback_category:
        return fallback_category

    return None

def _fallback_category_match(description: str) -> Optional[str]:
    """
    Fallback matching for descriptions that don't match standard patterns.
    Uses keyword analysis and fuzzy matching.

    Args:
        description: Payment description

    Returns:
        Category name or None if no fallback match
    """
    desc_lower = description.lower()
    words = desc_lower.split()

    # Insurance keywords
    insurance_keywords = ['versicherung', 'insurance', 'versicherungs', 'haftpflicht', 'kfz', 'kasko', 'auto', 'car', 'haus', 'wohnung', 'rechtsschutz', 'kranken', 'gesundheit']
    if any(keyword in desc_lower for keyword in insurance_keywords):
        # Try to determine specific insurance type
        if any(word in desc_lower for word in ['haftpflicht', 'haftpflichtversicherung']):
            return "Liability Insurance"
        elif any(word in desc_lower for word in ['kfz', 'kasko', 'auto', 'car']):
            return "Car Insurance"
        elif any(word in desc_lower for word in ['haus', 'wohnung', 'wohngebaeude']):
            return "Home Insurance"
        elif any(word in desc_lower for word in ['rechtsschutz']):
            return "Legal Protection"
        elif any(word in desc_lower for word in ['kranken', 'gesundheit']):
            return "Health Insurance"
        else:
            return "Liability Insurance"  # Default to most common

    # Telecom keywords
    telecom_keywords = ['telekom', 'vodafone', 'o2', 'mobil', 'handy', 'internet', 'dsl', 'telefon', 'phone', 'mobile']
    if any(keyword in desc_lower for keyword in telecom_keywords):
        if any(word in desc_lower for word in ['internet', 'dsl']):
            return "Internet"
        elif any(word in desc_lower for word in ['mobil', 'handy', 'telefon', 'phone', 'mobile']):
            return "Mobile"
        else:
            return "Internet"  # Default telecom

    # Utility keywords
    utility_keywords = ['strom', 'electricity', 'gas', 'wasser', 'water', 'stadtwerke', 'energie', 'energy']
    if any(keyword in desc_lower for keyword in utility_keywords):
        if any(word in desc_lower for word in ['strom', 'electricity', 'energie']):
            return "Electricity"
        elif any(word in desc_lower for word in ['gas']):
            return "Gas"
        elif any(word in desc_lower for word in ['wasser', 'water']):
            return "Water"
        else:
            return "Electricity"  # Default utility

    # Entertainment keywords
    entertainment_keywords = ['netflix', 'amazon', 'disney', 'spotify', 'music', 'streaming', 'entertainment', 'prime', 'plus']
    if any(keyword in desc_lower for keyword in entertainment_keywords):
        if any(word in desc_lower for word in ['spotify', 'music']):
            return "Music"
        else:
            return "Entertainment"

    # Software keywords
    software_keywords = ['microsoft', 'adobe', 'software', 'cloud', 'workspace', 'office', '365']
    if any(keyword in desc_lower for keyword in software_keywords):
        return "Software"

    # If we can't match anything, try to infer from the first word or common patterns
    first_word = words[0] if words else ""

    # Common provider name mappings
    provider_mappings = {
        'allianz': "Liability Insurance",
        'huk': "Car Insurance",
        'generali': "Home Insurance",
        'axa': "Legal Protection",
        'dkv': "Health Insurance",
        'telekom': "Internet",
        'vodafone': "Internet",
        'o2': "Mobile",
        'e.on': "Electricity",
        'rwe': "Electricity",
        'enbw': "Electricity",
        'stadtwerke': "Gas",
        'microsoft': "Software",
        'adobe': "Software",
        'netflix': "Entertainment",
        'amazon': "Entertainment",
        'spotify': "Music"
    }

    for provider, category in provider_mappings.items():
        if provider in desc_lower:
            return category

    return None

def get_category_insights(category: str) -> Dict:
    """
    Get insights about a specific category.

    Args:
        category: Category name

    Returns:
        Dictionary with category insights
    """
    if category not in ALTERNATIVE_PROVIDERS:
        return {}

    alternatives = ALTERNATIVE_PROVIDERS[category]
    avg_rating = sum(alt['rating'] for alt in alternatives) / len(alternatives)
    max_savings = max(alt['savings'] for alt in alternatives)
    avg_price = sum(alt['price'] for alt in alternatives) / len(alternatives)

    return {
        'category': category,
        'num_alternatives': len(alternatives),
        'average_rating': round(avg_rating, 1),
        'max_savings_percentage': round(max_savings, 1),
        'average_price': round(avg_price, 2)
    }
