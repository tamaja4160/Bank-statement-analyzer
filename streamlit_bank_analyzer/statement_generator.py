"""
Modified statement generator for Streamlit app
Generates bank statements with custom names and returns image paths.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import uuid

import excel2img
import pandas as pd
from faker import Faker
from openpyxl import load_workbook
from openpyxl.styles import Alignment

# Constants from the original generator (copied to avoid import issues)
RECURRING_PAYMENTS = [
    {
        "partner": "Allianz SE", "category": "Insurance",
        "description_template": "BEITRAG {partner} K-{contract_id}",
        "base_amount": 55.20
    },
    {
        "partner": "VODAFONE GMBH", "category": "Internet",
        "description_template": "RECHNUNG {partner} {contract_id}",
        "base_amount": 39.99
    },
    {
        "partner": "ZEUS BODYPOWER", "category": "Gym",
        "description_template": "MITGLIEDSBEITRAG {partner}",
        "base_amount": 25.00
    },
    {
        "partner": "Stadtwerke Rosenheim", "category": "Electricity",
        "description_template": "ABSCHLAG STROM {partner} {contract_id}",
        "base_amount": 85.50
    },
]

ONE_OFF_PAYMENTS = [
    {
        "partner_options": ["EDEKA", "REWE", "LIDL", "ALDI SUED"], "category": "Groceries",
        "description_template": "KARTENZ./{date_short} {partner} RO",
        "amount_range": (15.0, 150.0)
    },
    {
        "partner_options": ["AMAZON.DE", "ZALANDO", "EBAY"], "category": "Shopping",
        "description_template": "{partner} MKTPLC EU {random_str}",
        "amount_range": (10.0, 250.0)
    },
    {
        "partner_options": ["SHELL", "ARAL", "JET"], "category": "Fuel",
        "description_template": "KARTENZAHLUNG {partner} TANKSTELLE",
        "amount_range": (40.0, 90.0)
    },
    {
        "partner_options": ["BURGER KING", "MCDONALDS"], "category": "Dining",
        "description_template": "{partner} {city}",
        "amount_range": (8.0, 45.0)
    },
    {
       "partner_options": ["PAYPAL"], "category": "General",
       "description_template": "PAYPAL {random_str}",
       "amount_range": (5.0, 100.0)
    }
]

TRANSACTION_START_ROW = 6
TRANSACTION_END_ROW = 11
END_BALANCE_ROW = 12
NEXT_BILLING_ROW = 13
MAX_TRANSACTIONS_PER_STATEMENT = (TRANSACTION_END_ROW - TRANSACTION_START_ROW) + 1

def generate_statement_data(fake, statement_number):
    """Generate realistic transaction data for a statement."""
    transactions = []
    current_balance = 0.0
    start_date = fake.date_between(start_date="-2y", end_date="today")
    current_date = start_date

    num_total_transactions = random.randint(3, MAX_TRANSACTIONS_PER_STATEMENT)
    potential_recurring = random.sample(
        RECURRING_PAYMENTS,
        random.randint(1, min(len(RECURRING_PAYMENTS), num_total_transactions - 1))
    )

    transaction_pool = potential_recurring + [None] * (num_total_transactions - len(potential_recurring))
    random.shuffle(transaction_pool)

    for transaction_type in transaction_pool:
        current_date += timedelta(days=random.randint(1, 4))
        beleg_date = current_date.strftime("%d.%m.")
        valuta_date = (current_date + timedelta(days=1)).strftime("%d.%m.")

        if transaction_type:  # It's a recurring payment
            partner = transaction_type["partner"]
            description = transaction_type["description_template"].format(
                partner=partner,
                contract_id=fake.random_number(digits=8)
            )
            amount = round(transaction_type["base_amount"] + random.uniform(-2.5, 2.5), 2)
        else:  # It's a one-off payment
            one_off = random.choice(ONE_OFF_PAYMENTS)
            partner = random.choice(one_off["partner_options"])
            description = one_off["description_template"].format(
                date_short=current_date.strftime("%d.%m"),
                partner=partner,
                random_str=fake.lexify(text='??????').upper(),
                city=fake.city()
            )
            amount = round(random.uniform(*one_off["amount_range"]), 2)

        transactions.append([beleg_date, valuta_date, description, f"{amount:.2f}".replace('.', ',') + "-"])
        current_balance += amount

    end_date = current_date + timedelta(days=random.randint(2, 5))
    next_billing_date = end_date + timedelta(days=random.randint(5, 10))

    return start_date, end_date, next_billing_date, transactions, current_balance

def generate_statements_for_name(name: str, num_statements: int = 10) -> list:
    """
    Generate bank statements for a specific name and return image paths.

    Args:
        name: Full name to use for the statements
        num_statements: Number of statements to generate

    Returns:
        List of paths to generated PNG images
    """
    try:
        # Create session-specific output directory
        session_id = str(uuid.uuid4())[:8]
        output_dir = Path(f"streamlit_sessions/{session_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split name into first and last
        name_parts = name.strip().split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = ' '.join(name_parts[1:])
        else:
            first_name = name_parts[0]
            last_name = "Doe"  # Default last name

        # Generate fake data but use provided name
        fake = Faker('de_DE')

        person_data = {
            "first_name": first_name,
            "last_name": last_name,
            "card_number": fake.credit_card_number(card_type='mastercard'),
        }

        image_paths = []

        for i in range(1, num_statements + 1):
            excel_filename = output_dir / f"statement_{i}.xlsx"
            image_filename = output_dir / f"statement_{i}.png"

            # Generate statement data
            start_date, end_date, next_billing_date, transactions, final_balance = generate_statement_data(fake, i)

            # Create Excel file
            template_path = os.path.join(os.path.dirname(__file__), "..", "bank statement generator", "template.xlsx")
            wb = load_workbook(template_path)
            ws = wb.active

            # Fill header with consistent person data
            ws["B2"] = "MasterCard"
            ws["B3"] = person_data["card_number"]
            ws["C2"] = person_data["first_name"]
            ws["C3"] = person_data["last_name"]
            ws["F2"] = i
            ws["F3"] = 1

            # Fill fixed position balances and dates
            ws[f"D{5}"] = f"KONTOSTAND AM {start_date.strftime('%d.%m.%Y')}"
            ws[f"E{5}"].alignment = Alignment(horizontal='right')

            ws[f"D{12}"] = f"KONTOSTAND AM {end_date.strftime('%d.%m.%Y')}"
            ws[f"E{12}"] = f"{final_balance:.2f}".replace('.', ',') + "-"
            ws[f"E{12}"].alignment = Alignment(horizontal='right')

            ws[f"C{13}"] = f"IHR NAECHSTER ABRECHNUNGSTERMIN {next_billing_date.strftime('%d.%m.%Y')}"

            # Fill transaction rows
            for j, trx_data in enumerate(transactions):
                row = TRANSACTION_START_ROW + j
                ws[f'B{row}'], ws[f'C{row}'], ws[f'D{row}'], ws[f'E{row}'] = trx_data
                ws[f'E{row}'].alignment = Alignment(horizontal='right')

            # Clear any unused transaction rows
            for j in range(len(transactions), MAX_TRANSACTIONS_PER_STATEMENT):
                row_to_clear = TRANSACTION_START_ROW + j
                ws[f'B{row_to_clear}'].value, ws[f'C{row_to_clear}'].value, ws[f'D{row_to_clear}'].value, ws[f'E{row_to_clear}'].value = (None, None, None, None)

            wb.save(excel_filename)

            # Convert to image
            try:
                image_range = f"A1:F{NEXT_BILLING_ROW + 1}"
                excel2img.export_img(str(excel_filename), str(image_filename), wb.active.title, image_range)
                image_paths.append(str(image_filename))
            except Exception as e:
                print(f"Error converting {excel_filename} to image: {e}")
                continue

        return image_paths

    except Exception as e:
        print(f"Error generating statements: {e}")
        return []
