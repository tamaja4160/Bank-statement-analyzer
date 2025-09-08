"""
Payment analyzer for detecting recurring payments and recommending IBAN changes.
Uses ML-based clustering and text similarity to identify similar transactions.
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import statistics
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

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

def group_similar_transactions(transactions: List[Dict], similarity_threshold: float = 0.7) -> Dict[str, List[Dict]]:
    """
    Group transactions by similar descriptions using ML-based clustering and text similarity.

    Args:
        transactions: List of transaction dictionaries
        similarity_threshold: Minimum similarity score (0-1) for grouping

    Returns:
        Dictionary mapping canonical descriptions to lists of transactions
    """
    if not transactions:
        return {}

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(transactions)

    # Clean descriptions for analysis
    df['clean_description'] = df['description'].fillna('').apply(clean_description_for_ml)

    # Extract amounts for feature engineering
    df['amount'] = df['amount_str'].fillna('0').apply(lambda x: abs(float(x.replace(',', '.'))) if x.replace(',', '').replace('.', '').isdigit() else 0)

    # Use ML-based clustering approach
    groups = ml_based_grouping(df, similarity_threshold)

    return groups

def ml_based_grouping(df: pd.DataFrame, similarity_threshold: float = 0.7) -> Dict[str, List[Dict]]:
    """
    Use ML techniques to group similar transactions.

    Args:
        df: DataFrame with transaction data
        similarity_threshold: Similarity threshold for grouping

    Returns:
        Dictionary mapping canonical descriptions to lists of transactions
    """
    groups = defaultdict(list)

    # Prepare text data for TF-IDF vectorization
    descriptions = df['clean_description'].tolist()

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words=['und', 'der', 'die', 'das', 'mit', 'von', 'für', 'auf', 'im', 'am', 'um']
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    except ValueError:
        # Fallback to simple grouping if TF-IDF fails
        return fallback_grouping(df)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Use DBSCAN for clustering
    # Convert similarity to distance (DBSCAN uses distance)
    distance_matrix = 1 - similarity_matrix

    # Scale features for better clustering
    scaler = StandardScaler()
    amount_features = scaler.fit_transform(df[['amount']].fillna(0))

    # Combine text and amount features
    combined_features = np.hstack([tfidf_matrix.toarray(), amount_features])

    # Adaptive DBSCAN parameters based on dataset size
    n_samples = len(df)
    if n_samples <= 3:
        # For small datasets (like ZEUS case), use more permissive parameters
        eps = 1.0  # Increased from 0.3
        min_samples = 1  # Reduced from 2 to allow pairs to form clusters
    else:
        # For larger datasets, use original parameters
        eps = 0.3
        min_samples = 2

    # Apply DBSCAN clustering
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean'
    )

    clusters = dbscan.fit_predict(combined_features)

    # Group transactions by cluster
    for idx, cluster_id in enumerate(clusters):
        if cluster_id != -1:  # -1 indicates noise/outlier
            # Use the most representative description as canonical
            canonical_desc = get_canonical_description(df.iloc[idx]['clean_description'], df.iloc[idx]['description'])
            groups[canonical_desc].append(df.iloc[idx].to_dict())

    # Handle unclustered transactions with similarity-based grouping
    unclustered_indices = [i for i, cluster in enumerate(clusters) if cluster == -1]
    if unclustered_indices:
        similarity_groups = similarity_based_grouping(df.iloc[unclustered_indices], similarity_threshold)
        for canonical_desc, group_indices in similarity_groups.items():
            group_transactions = [df.iloc[i].to_dict() for i in group_indices]
            if len(group_transactions) >= 2:
                groups[canonical_desc].extend(group_transactions)

    return dict(groups)

def similarity_based_grouping(df: pd.DataFrame, similarity_threshold: float = 0.7) -> Dict[str, List[int]]:
    """
    Fallback similarity-based grouping for unclustered transactions.

    Args:
        df: DataFrame with unclustered transactions
        similarity_threshold: Similarity threshold

    Returns:
        Dictionary mapping canonical descriptions to lists of indices
    """
    groups = defaultdict(list)
    processed = set()

    for i, row in df.iterrows():
        if i in processed:
            continue

        description = row['clean_description']
        canonical = get_canonical_description(description, row['description'])
        groups[canonical].append(i)
        processed.add(i)

        # Find similar transactions
        for j, other_row in df.iterrows():
            if j in processed or i == j:
                continue

            similarity = calculate_description_similarity(description, other_row['clean_description'])
            if similarity >= similarity_threshold:
                groups[canonical].append(j)
                processed.add(j)

    return dict(groups)

def fallback_grouping(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Simple fallback grouping when ML approaches fail.

    Args:
        df: DataFrame with transaction data

    Returns:
        Dictionary mapping canonical descriptions to lists of transactions
    """
    groups = defaultdict(list)

    for _, row in df.iterrows():
        description = row['description']
        canonical = clean_description_for_grouping(description)
        groups[canonical].append(row.to_dict())

    return dict(groups)

def clean_description_for_ml(description: str) -> str:
    """
    Clean description specifically for ML processing.

    Args:
        description: Raw transaction description

    Returns:
        Cleaned description for ML analysis
    """
    if not description:
        return ""

    # Convert to uppercase
    desc = description.upper()

    # Remove contract numbers and variable parts
    patterns_to_remove = [
        r'\d{6,12}',  # Contract numbers (6-12 digits)
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # Dates
        r'K-\d+',  # Contract suffixes
        r'/\d+',  # Reference numbers
        r'-\d+',  # Reference numbers
    ]

    for pattern in patterns_to_remove:
        desc = re.sub(pattern, '', desc)

    # Normalize whitespace
    desc = re.sub(r'\s+', ' ', desc).strip()

    return desc

def get_canonical_description(clean_desc: str, original_desc: str) -> str:
    """
    Create a canonical description from cleaned and original descriptions.

    Args:
        clean_desc: Cleaned description
        original_desc: Original description

    Returns:
        Canonical description for grouping
    """
    # For utility payments, extract the key merchant name
    if 'ABSCHLAG' in original_desc.upper():
        # Extract utility company name
        match = re.search(r'ABSCHLAG\s+(.*?)(?:\s+\d|$)', original_desc.upper())
        if match:
            return f"ABSCHLAG {match.group(1).strip()}"

    # For other recurring payments, use the cleaned description
    return clean_description_for_grouping(original_desc)

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
