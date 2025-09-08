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
    clean_description
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
        cleaned = clean_description(tx['description'])
        print(f"  Original: '{tx['description']}'")
        print(f"  Cleaned:  '{cleaned}'")

    print("\n📋 Full Analysis Result:")
    results = analyze_recurring_payments(test_transactions)
    print(f"  Total analyzed: {results['total_analyzed']}")
    print(f"  Recurring payments found: {len(results['recurring_payments'])}")
    if results['recurring_payments']:
        for payment in results['recurring_payments']:
            print(f"  ✅ {payment['description']} - {payment['occurrences']} occurrences")
    else:
        print("  ❌ No recurring payments detected")

def debug_with_real_data():
    """Debug with the actual ZEUS transactions from the CSV file."""
    print("\n🔍 Testing with REAL ZEUS BODYPOWER Data")
    print("=" * 60)

    # Load the actual CSV data
    csv_path = 'all_extracted_transactions.csv'
    if not os.path.isfile(csv_path):
        print(f"⚠️ CSV file '{csv_path}' not found. Skipping real data debug.")
        return
    df = pd.read_csv(csv_path)

    # Filter for ZEUS BODYPOWER transactions
    zeus_transactions = []
    for _, row in df.iterrows():
        if 'MITGLIEDSBEITRAG ZEUS BODYPOWER' in str(row['description']):
            zeus_transactions.append({
                'description': row['description'],
                'amount_str': str(row['amount_str']),
                'formatted_date': row['formatted_date']
            })

    print(f"Found {len(zeus_transactions)} ZEUS BODYPOWER transactions:")
    for i, tx in enumerate(zeus_transactions[:10], 1):  # Show first 10
        print(f"  {i}. {tx['description']} - {tx['amount_str']}€ on {tx['formatted_date']}")

    if len(zeus_transactions) > 10:
        print(f"  ... and {len(zeus_transactions) - 10} more")

    print("\n🧹 Testing with ALL ZEUS transactions:")
    results = analyze_recurring_payments(zeus_transactions)
    print(f"  Total analyzed: {results['total_analyzed']}")
    print(f"  Recurring payments found: {len(results['recurring_payments'])}")

    if results['recurring_payments']:
        for payment in results['recurring_payments']:
            print(f"  ✅ {payment['description']} - {payment['occurrences']} occurrences")
            print(f"     Average amount: €{payment['average_amount']:.2f}")
    else:
        print("  ❌ No recurring payments detected")

def debug_vodafone_case():
    """Debug Vodafone recurring payments."""
    print("\n📱 Testing Vodafone Case")
    print("=" * 40)

    # Test Vodafone transactions
    vodafone_transactions = [
        {
            'description': 'RECHNUNG VODAFONE GMBH 12345678',
            'amount_str': '39,99',
            'formatted_date': '15.01.2025'
        },
        {
            'description': 'RECHNUNG VODAFONE GMBH 87654321',
            'amount_str': '39,99',
            'formatted_date': '15.02.2025'
        },
        {
            'description': 'RECHNUNG VODAFONE GMBH 11223344',
            'amount_str': '39,99',
            'formatted_date': '15.03.2025'
        }
    ]

    print("📊 Vodafone Test Transactions:")
    for i, tx in enumerate(vodafone_transactions, 1):
        print(f"  {i}. '{tx['description']}' - {tx['amount_str']}€ on {tx['formatted_date']}")

    print("\n🧹 Text Cleaning Analysis:")
    for tx in vodafone_transactions:
        cleaned = clean_description(tx['description'])
        print(f"  Original: '{tx['description']}'")
        print(f"  Cleaned:  '{cleaned}'")

    print("\n📋 Vodafone Analysis Result:")
    results = analyze_recurring_payments(vodafone_transactions)
    print(f"  Total analyzed: {results['total_analyzed']}")
    print(f"  Recurring payments found: {len(results['recurring_payments'])}")
    if results['recurring_payments']:
        for payment in results['recurring_payments']:
            print(f"  ✅ {payment['description']} - {payment['occurrences']} occurrences")
            print(f"     Average amount: €{payment['average_amount']:.2f}")
    else:
        print("  ❌ No recurring payments detected")

if __name__ == "__main__":
    debug_zeus_case()
    debug_vodafone_case()
    debug_with_real_data()
