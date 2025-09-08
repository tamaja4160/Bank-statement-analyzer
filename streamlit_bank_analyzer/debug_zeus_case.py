#!/usr/bin/env python3
"""
Debug script for the ZEUS BODYPOWER membership fee case.
Tests why "Mitgliedsbeitrag ZEUS BODYPOWER" isn't being detected as recurring.
"""

import sys
import os
sys.path.append('.')

from payment_analyzer import (
    analyze_recurring_payments,
    group_similar_transactions,
    analyze_payment_pattern,
    clean_description_for_ml,
    calculate_description_similarity
)
from advanced_ml import AdvancedMLProcessor
import pandas as pd

def debug_zeus_case():
    """Debug the ZEUS BODYPOWER membership fee case."""
    print("🔍 Debugging ZEUS BODYPOWER Case")
    print("=" * 50)

    # Your exact test case
    test_transactions = [
        {
            'description': 'Mitgliedsbeitrag ZEUS BODYPOWER',
            'amount_str': '24,12',
            'formatted_date': '10.02.2024'
        },
        {
            'description': 'Mitgliedsbeitrag ZEUS BODYPOWER',
            'amount_str': '23,02',
            'formatted_date': '06.08.2024'
        }
    ]

    print("📊 Test Transactions:")
    for i, tx in enumerate(test_transactions, 1):
        print(f"  {i}. '{tx['description']}' - {tx['amount_str']}€ on {tx['formatted_date']}")

    print("\n🧹 Text Cleaning Analysis:")
    for tx in test_transactions:
        cleaned = clean_description_for_ml(tx['description'])
        print(f"  Original: '{tx['description']}'")
        print(f"  Cleaned:  '{cleaned}'")

    print("\n📏 Similarity Analysis:")
    desc1 = clean_description_for_ml(test_transactions[0]['description'])
    desc2 = clean_description_for_ml(test_transactions[1]['description'])
    similarity = calculate_description_similarity(desc1, desc2)
    print(f"  Similarity between cleaned descriptions: {similarity:.3f}")

    print("\n🔗 Grouping Analysis:")
    groups = group_similar_transactions(test_transactions, similarity_threshold=0.7)
    print(f"  Number of groups: {len(groups)}")
    for group_name, group_txs in groups.items():
        print(f"  Group '{group_name}': {len(group_txs)} transactions")
        for tx in group_txs:
            print(f"    - {tx['description']} ({tx['amount_str']}€)")

    print("\n🎯 Pattern Analysis:")
    if groups:
        for group_name, group_txs in groups.items():
            if len(group_txs) >= 2:
                pattern_result = analyze_payment_pattern(group_txs)
                print(f"  Group '{group_name}':")
                print(f"    - Is recurring: {pattern_result['is_recurring']}")
                print(f"    - Confidence: {pattern_result['confidence']:.3f}")
                print(f"    - Pattern type: {pattern_result['pattern_type']}")
                print(f"    - Amount similarity: {pattern_result['amount_similarity']:.3f}")
                print(f"    - Amounts: {pattern_result['amounts']}")

    print("\n🤖 ML Analysis:")
    ml_processor = AdvancedMLProcessor()
    features_df = ml_processor.extract_transaction_features(test_transactions)
    print(f"  Features extracted: {len(features_df.columns)}")
    print(f"  Sample features: {list(features_df.columns[:10])}")

    # Test clustering with different parameters
    print("\n📈 Clustering Analysis:")
    clusters = ml_processor.cluster_similar_transactions(test_transactions)
    print(f"  Clusters found: {len(clusters)}")
    for cluster_name, cluster_txs in clusters.items():
        print(f"  {cluster_name}: {len(cluster_txs)} transactions")

    print("\n🔧 Parameter Testing:")
    # Test with different DBSCAN eps values
    for eps in [0.1, 0.3, 0.5, 0.8]:
        test_clusters = ml_processor._test_clustering_eps(test_transactions, eps)
        print(f"  eps={eps}: {len(test_clusters)} clusters")

    print("\n📋 Full Analysis Result:")
    results = analyze_recurring_payments(test_transactions)
    print(f"  Total analyzed: {results['total_analyzed']}")
    print(f"  Recurring payments found: {len(results['recurring_payments'])}")
    if results['recurring_payments']:
        for payment in results['recurring_payments']:
            print(f"  ✅ {payment['description']} - {payment['occurrences']} occurrences")
    else:
        print("  ❌ No recurring payments detected")

def debug_clustering_details():
    """Debug the clustering process in detail."""
    print("\n🔬 Detailed Clustering Debug")
    print("=" * 50)

    test_transactions = [
        {
            'description': 'Mitgliedsbeitrag ZEUS BODYPOWER',
            'amount_str': '24,12',
            'formatted_date': '10.02.2024'
        },
        {
            'description': 'Mitgliedsbeitrag ZEUS BODYPOWER',
            'amount_str': '23,02',
            'formatted_date': '06.08.2024'
        }
    ]

    ml_processor = AdvancedMLProcessor()

    # Extract features
    df = ml_processor.extract_transaction_features(test_transactions)
    print("Feature matrix:")
    print(df)

    # Add clean_description column for compatibility
    df['clean_description'] = df['original_description'].apply(clean_description_for_ml)

    # Test TF-IDF
    descriptions = df['clean_description'].tolist()
    print(f"\nDescriptions: {descriptions}")

    # Manual similarity calculation
    if len(descriptions) == 2:
        sim = calculate_description_similarity(descriptions[0], descriptions[1])
        print(f"Manual similarity: {sim:.3f}")

    # Test different clustering approaches
    print("\nTesting different clustering approaches:")

    # 1. Simple similarity-based
    simple_groups = {}
    for i, tx in enumerate(test_transactions):
        desc = clean_description_for_ml(tx['description'])
        if desc not in simple_groups:
            simple_groups[desc] = []
        simple_groups[desc].append(tx)

    print(f"Simple grouping: {len(simple_groups)} groups")

    # 2. Amount-based clustering
    amounts = [float(tx['amount_str'].replace(',', '.')) for tx in test_transactions]
    amount_diff = abs(amounts[0] - amounts[1])
    amount_similarity = amount_diff / max(amounts) if max(amounts) > 0 else 1.0
    print(f"Amount difference: {amount_diff:.2f}€")
    print(f"Amount similarity: {amount_similarity:.3f}")

    # 3. Debug DBSCAN parameters
    print("\n🔧 DBSCAN Parameter Analysis:")
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Prepare features for clustering
    feature_cols = [col for col in df.columns if col not in ['original_description', 'original_amount', 'clean_description']]
    X = df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Feature matrix sample:\n{X_scaled}")

    # Test different DBSCAN parameters
    eps_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    min_samples_values = [1, 2]

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            clusters = dbscan.fit_predict(X_scaled)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points")

if __name__ == "__main__":
    debug_zeus_case()
    debug_clustering_details()
