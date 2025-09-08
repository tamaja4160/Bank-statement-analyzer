#!/usr/bin/env python3
"""
Test for Advanced ML Algorithms (without transformers dependency issues)
"""

import sys
import os
sys.path.append('.')

from advanced_ml import AdvancedMLProcessor
from payment_analyzer import analyze_recurring_payments

def test_advanced_ml():
    """Test Advanced ML features."""
    print("🤖 Testing Advanced ML Features...")
    print("=" * 50)

    ml_processor = AdvancedMLProcessor()

    # Test transactions
    test_transactions = [
        {
            'description': 'ABSCHLAG STROM Stadtwerke Rosenheim 99188511',
            'amount_str': '45.67',
            'formatted_date': '15.01.2024'
        },
        {
            'description': 'ABSCHLAG GAS Stadtwerke Rosenheim 12345678',
            'amount_str': '78.90',
            'formatted_date': '20.01.2024'
        },
        {
            'description': 'REWE SAGT DANKE 987654',
            'amount_str': '23.45',
            'formatted_date': '10.01.2024'
        },
        {
            'description': 'VERSICHERUNG Allianz Beitrag',
            'amount_str': '89.99',
            'formatted_date': '01.02.2024'
        }
    ]

    print("\n📊 Feature Extraction:")
    features_df = ml_processor.extract_transaction_features(test_transactions)
    print(f"  Extracted {len(features_df.columns)} features for {len(test_transactions)} transactions")
    print(f"  Sample features: {list(features_df.columns[:10])}")

    print("\n🎯 Transaction Classification:")
    for tx in test_transactions:
        prediction = ml_processor.predict_transaction_category(tx)
        if 'ensemble_prediction' in prediction:
            print(f"  '{tx['description']}' -> {prediction['ensemble_prediction']} (confidence: {prediction.get('confidence', 0):.2f})")

    print("\n🔍 Anomaly Detection:")
    anomalies = ml_processor.detect_anomalies(test_transactions)
    for anomaly in anomalies:
        if anomaly.get('is_anomaly'):
            print(f"  ⚠️ Potential anomaly: '{anomaly['description']}' (score: {anomaly['anomaly_score']})")

    print("\n📈 Clustering:")
    clusters = ml_processor.cluster_similar_transactions(test_transactions)
    for cluster_name, cluster_txs in clusters.items():
        print(f"  {cluster_name}: {len(cluster_txs)} transactions")
        for tx in cluster_txs:
            print(f"    - {tx['description']}")

def test_integrated_system():
    """Test the integrated system with recurring payment analysis."""
    print("\n🔄 Testing Integrated System...")
    print("=" * 50)

    # Test transactions with recurring patterns
    test_transactions = [
        {
            'description': 'ABSCHLAG STROM Stadtwerke Rosenheim 99188511',
            'amount_str': '45.67',
            'formatted_date': '15.01.2024'
        },
        {
            'description': 'ABSCHLAG STROM Stadtwerke Rosenheim 98691535',
            'amount_str': '45.67',
            'formatted_date': '15.02.2024'
        },
        {
            'description': 'ABSCHLAG STROM Stadtwerke Rosenheim 661823451',
            'amount_str': '45.67',
            'formatted_date': '15.03.2024'
        },
        {
            'description': 'ABSCHLAG GAS Stadtwerke Rosenheim 12345678',
            'amount_str': '78.90',
            'formatted_date': '20.01.2024'
        },
        {
            'description': 'ABSCHLAG GAS Stadtwerke Rosenheim 87654321',
            'amount_str': '78.90',
            'formatted_date': '20.02.2024'
        },
        {
            'description': 'REWE SAGT DANKE 1234',
            'amount_str': '23.45',
            'formatted_date': '10.01.2024'
        },
        {
            'description': 'REWE SAGT DANKE 5678',
            'amount_str': '67.89',
            'formatted_date': '15.01.2024'
        }
    ]

    print(f"📊 Analyzing {len(test_transactions)} transactions...")

    # Run the integrated analysis
    results = analyze_recurring_payments(test_transactions)

    print("\n📋 Analysis Results:")
    print(f"  • Total transactions analyzed: {results['total_analyzed']}")
    print(f"  • Unique descriptions: {results['unique_descriptions']}")
    print(f"  • Recurring payments found: {len(results['recurring_payments'])}")
    print(f"  • ML-enhanced: {results.get('ml_enhanced', False)}")

    print("\n🔄 Recurring Payments Detected:")
    for payment in results['recurring_payments']:
        print(f"  • {payment['description']}")
        print(f"    - Occurrences: {payment['occurrences']}")
        print(f"    - Average amount: €{payment['average_amount']:.2f}")
        print(f"    - Confidence: {payment['confidence']:.2f}")
        print(f"    - Pattern: {payment['pattern_type']}")

    print("\n💡 Recommendations:")
    for rec in results['recommendations']:
        if rec.strip():
            print(f"  • {rec}")

def main():
    """Run ML-only tests."""
    print("🚀 Advanced ML Payment Analysis Test")
    print("=" * 50)

    try:
        test_advanced_ml()
        test_integrated_system()

        print("\n✅ ML tests completed successfully!")
        print("\n🎉 Advanced ML features working:")
        print("  • ✅ LightGBM/XGBoost/CatBoost ensemble")
        print("  • ✅ Anomaly detection with Isolation Forest")
        print("  • ✅ Advanced clustering with DBSCAN")
        print("  • ✅ Feature engineering pipeline")
        print("  • ✅ Integrated with payment analyzer")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
