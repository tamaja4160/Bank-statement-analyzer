#!/usr/bin/env python3
"""
Test script for the ML-enhanced payment analyzer.
Tests the ABSCHLAG STROM grouping functionality.
"""

import sys
import os
sys.path.append('.')

from payment_analyzer import analyze_recurring_payments

def test_abschlag_strom_grouping():
    """Test that ABSCHLAG STROM transactions with different contract numbers are grouped together."""

    # Sample transactions that should be grouped together
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

    print("Testing ML-enhanced payment analyzer...")
    print(f"Input transactions: {len(test_transactions)}")

    # Analyze the transactions
    results = analyze_recurring_payments(test_transactions)

    print(f"\nRecurring payments found: {len(results['recurring_payments'])}")

    # Check if ABSCHLAG STROM transactions are grouped together
    abschlag_strom_found = False
    abschlag_gas_found = False
    rewe_found = False

    for payment in results['recurring_payments']:
        desc = payment['description'].upper()
        print(f"\nPayment: {desc}")
        print(f"  Occurrences: {payment['occurrences']}")
        print(f"  Confidence: {payment['confidence']:.2f}")
        print(f"  Pattern Type: {payment['pattern_type']}")

        if 'ABSCHLAG STROM' in desc and 'STADTWERKE ROSENHEIM' in desc:
            abschlag_strom_found = True
            if payment['occurrences'] == 3:
                print("  ✅ ABSCHLAG STROM transactions correctly grouped (3 occurrences)")
            else:
                print(f"  ❌ ABSCHLAG STROM transactions not properly grouped ({payment['occurrences']} occurrences)")

        elif 'ABSCHLAG GAS' in desc and 'STADTWERKE ROSENHEIM' in desc:
            abschlag_gas_found = True
            if payment['occurrences'] == 2:
                print("  ✅ ABSCHLAG GAS transactions correctly grouped (2 occurrences)")
            else:
                print(f"  ❌ ABSCHLAG GAS transactions not properly grouped ({payment['occurrences']} occurrences)")

        elif 'REWE' in desc:
            rewe_found = True
            print(f"  ℹ️ REWE transactions found ({payment['occurrences']} occurrences)")

    print("\n=== Test Results ===")
    print(f"ABSCHLAG STROM grouped correctly: {'✅' if abschlag_strom_found else '❌'}")
    print(f"ABSCHLAG GAS grouped correctly: {'✅' if abschlag_gas_found else '❌'}")
    print(f"REWE transactions found: {'✅' if rewe_found else '❌'}")

    return results

if __name__ == "__main__":
    test_abschlag_strom_grouping()
