#!/usr/bin/env python3
"""
Test script for deal comparison functionality.
Tests the category matching and deal finding logic.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from deal_comparison import find_better_deals, _match_payment_to_category

def test_category_matching():
    """Test the category matching functionality."""
    print("Testing category matching...")

    test_cases = [
        # Insurance
        ("HAFTPFLICHT ALLIANZ SE K-12345678", "Liability Insurance"),
        ("KFZ-VERSICHERUNG HUK-COBURG 12345678", "Car Insurance"),
        ("WOHNGEBAEUDEVERSICHERUNG GENERALI 12345678", "Home Insurance"),
        ("RECHTSSCHUTZ AXA 12345678", "Legal Protection"),
        ("GESUNDHEITSVERSICHERUNG DKV DEUTSCHE KRANK 12345678", "Health Insurance"),

        # Telecom
        ("TELEKOM RECHNUNG 12345678", "Internet"),
        ("VODAFONE INTERNET 12345678", "Internet"),
        ("O2 MOBILFUNK 12345678", "Mobile"),
        ("1&1 MOBILE 12345678", "Mobile"),

        # Utilities
        ("E.ON STROM 12345678", "Electricity"),
        ("GASABSCHLAG STADTWERKE BERLIN 12345678", "Gas"),
        ("WASSERABSCHLAG STADTWERKE HAMBURG 12345678", "Water"),

        # Entertainment
        ("NETFLIX ABONNEMENT 12345678", "Entertainment"),
        ("AMAZON PRIME 12345678", "Entertainment"),
        ("SPOTIFY PREMIUM 12345678", "Music"),

        # Software
        ("MICROSOFT 365 12345678", "Software"),
        ("ADOBE CREATIVE CLOUD 12345678", "Software"),
    ]

    passed = 0
    failed = 0

    for description, expected_category in test_cases:
        result = _match_payment_to_category(description)
        if result == expected_category:
            print(f"✅ {description} -> {result}")
            passed += 1
        else:
            print(f"❌ {description} -> {result} (expected {expected_category})")
            failed += 1

    print(f"\nCategory matching: {passed} passed, {failed} failed")
    return failed == 0

def test_deal_finding():
    """Test the deal finding functionality."""
    print("\nTesting deal finding...")

    # Sample recurring payments that should match categories
    recurring_payments = [
        {
            'description': 'HAFTPFLICHT ALLIANZ SE K-12345678',
            'average_amount': 55.20,
            'occurrences': 3,
            'confidence': 0.85,
            'pattern_type': 'Regular Recurring',
            'amount_range': 2.5
        },
        {
            'description': 'TELEKOM RECHNUNG 12345678',
            'average_amount': 49.95,
            'occurrences': 5,
            'confidence': 0.90,
            'pattern_type': 'Regular Recurring',
            'amount_range': 0.0
        },
        {
            'description': 'E.ON STROM 12345678',
            'average_amount': 78.40,
            'occurrences': 4,
            'confidence': 0.88,
            'pattern_type': 'Regular Recurring',
            'amount_range': 4.0
        },
        {
            'description': 'NETFLIX ABONNEMENT 12345678',
            'average_amount': 15.99,
            'occurrences': 6,
            'confidence': 0.92,
            'pattern_type': 'Subscription/Service',
            'amount_range': 0.0
        }
    ]

    result = find_better_deals(recurring_payments)

    print(f"Found {len(result['deal_comparisons'])} deal comparisons")
    print(f"Total current cost: €{result['total_current_cost']:.2f}")
    print(f"Total monthly savings: €{result['total_monthly_savings']:.2f}")
    print(f"Total annual savings: €{result['total_annual_savings']:.2f}")
    print(f"Savings percentage: {result['savings_percentage']:.1f}%")

    # Check if we found deals
    if result['deal_comparisons']:
        print("\nDeal comparisons found:")
        for deal in result['deal_comparisons']:
            print(f"  {deal['current_provider']} -> {deal['alternative_provider']}: Save €{deal['monthly_savings']:.2f}/month")

        return True
    else:
        print("No deal comparisons found!")
        return False

def main():
    """Run all tests."""
    print("Testing Deal Comparison Functionality")
    print("=" * 50)

    matching_ok = test_category_matching()
    deals_ok = test_deal_finding()

    print("\n" + "=" * 50)
    if matching_ok and deals_ok:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
