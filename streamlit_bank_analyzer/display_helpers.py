"""
Display helper functions for the Streamlit bank analyzer app.
Handles UI components for showing statements, transactions, and analysis results.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List
from pathlib import Path

def apply_global_styles(table_font_px: int = 18):
    """
    Inject global CSS to enlarge fonts inside tables and dataframes across the app.

    Args:
        table_font_px: Target font size in pixels for table contents.
    """
    st.markdown(f"""
    <style>
    /* Make Streamlit's interactive DataFrame (st.dataframe) text larger */
    div[data-testid="stDataFrame"] * {{
        font-size: {table_font_px}px !important;
        line-height: 1.4 !important;
    }}

    /* Make static tables (st.table) text larger */
    table tbody td, table thead th {{
        font-size: {table_font_px}px !important;
        line-height: 1.4 !important;
    }}

    /* Enlarge expander headers that often wrap tables */
    div[role="button"][data-baseweb="accordion"] p {{
        font-size: {table_font_px}px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def display_statement_with_extractions(image_path: str, result: Dict):
    """
    Display a bank statement image alongside its extracted transaction lines.

    Args:
        image_path: Path to the statement image
        result: OCR processing result dictionary
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display the bank statement image
        try:
            st.image(image_path, caption=f"Bank Statement", use_column_width=True)
        except Exception as e:
            st.error(f"Could not display image: {e}")

    with col2:
        # Display extraction results
        st.subheader("📝 Extracted Transactions")

        if result.get('success', False):
            transactions = result.get('transactions', [])

            if transactions:
                # Display OCR confidence
                confidence = result.get('ocr_confidence', 0)
                confidence_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
                st.metric("OCR Confidence", f"{confidence:.1f}%", delta=None)

                # Display transactions in a table format instead of nested expanders
                transaction_data = []
                for i, tx in enumerate(transactions, 1):
                    transaction_data.append({
                        "ID": i,
                        "Date": tx.get('formatted_date', 'N/A'),
                        "Description": tx.get('description', 'N/A')[:30] + "..." if len(tx.get('description', '')) > 30 else tx.get('description', 'N/A'),
                        "Amount": f"€{tx.get('amount_value', 0):.2f}"
                    })

                if transaction_data:
                    st.dataframe(transaction_data, use_container_width=True)
            else:
                st.warning("No transactions extracted from this statement")
        else:
            st.error("❌ Failed to extract transactions")
            if 'error' in result:
                st.text(f"Error: {result['error']}")

def display_transaction_details(transaction: Dict):
    """
    Display detailed information for a single transaction.

    Args:
        transaction: Transaction dictionary from OCR extraction
    """
    # Create a clean display of transaction data
    details = {
        "Date": transaction.get('formatted_date', 'N/A'),
        "Description": transaction.get('description', 'N/A'),
        "Amount": f"€{transaction.get('amount_value', 0):.2f}",
        "Raw Line": transaction.get('raw_line', 'N/A')
    }

    # Display in a structured format
    for key, value in details.items():
        st.write(f"**{key}:** {value}")

def display_analysis_results(analysis_results: Dict, all_transactions: List[Dict]):
    """
    Display the recurring payment analysis results.

    Args:
        analysis_results: Results from the payment analyzer
        all_transactions: All extracted transactions for context
    """
    if not analysis_results:
        st.warning("No analysis results available")
        return

    # Recurring payments table
    recurring_payments = analysis_results.get('recurring_payments', [])

    if recurring_payments:

        # Create a dataframe for better display
        recurring_df = pd.DataFrame([
            {
                'Biller': payment['description'],
                'Occurrences': payment['occurrences'],
                'Avg Amount (€)': f"€{payment['average_amount']:.2f}"
            }
            for payment in recurring_payments
        ])

        # Display in a narrower, fixed-size table to prevent width jitter
        mid_col = st.columns([1, 2, 1])[1]
        with mid_col:
            st.dataframe(
                recurring_df,
                use_container_width=False,
                width=900,
                height=300,
                hide_index=True,
            )

        # Deal analysis summary
        deal_analysis = analysis_results.get('deal_analysis', {})
        if deal_analysis and deal_analysis.get('total_monthly_savings', 0) > 0:
            st.success(f"💰 **Potential Monthly Savings: €{deal_analysis['total_monthly_savings']:.2f}** "
                      f"({deal_analysis['savings_percentage']:.1f}% of recurring costs)")
            st.metric("Current Monthly Cost", f"€{deal_analysis['total_current_cost']:.2f}")

        # Detailed view of each recurring payment
        for payment in recurring_payments:
            with st.expander(f"📋 {payment['description']} - {payment['occurrences']} occurrences", expanded=False):
                display_recurring_payment_details(payment)


def display_recurring_payment_details(payment: Dict):
    """
    Display detailed information about a recurring payment.

    Args:
        payment: Recurring payment analysis dictionary
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Amount", f"€{payment['average_amount']:.2f}")
        st.metric("Amount Range", f"€{payment['amount_range']:.2f}")
        st.metric("Confidence Score", f"{payment['confidence']:.1%}")

    with col2:
        st.metric("Pattern Type", payment['pattern_type'])
        st.metric("Total Occurrences", payment['occurrences'])

    # Show individual transaction amounts
    amounts = payment.get('transactions', [])
    if amounts:
        amount_values = [tx.get('amount_value', 0) for tx in amounts]
        st.write("**Individual Amounts:**")
        st.write(", ".join([f"€{amt:.2f}" for amt in amount_values]))

def display_payment_insights(recurring_payments: List[Dict], all_transactions: List[Dict]):
    """
    Display additional insights about the payment patterns.

    Args:
        recurring_payments: List of recurring payment analyses
        all_transactions: All extracted transactions
    """
    st.subheader("🔍 Payment Insights")

    # Calculate some statistics
    total_recurring_amount = sum(p['average_amount'] * p['occurrences'] for p in recurring_payments)
    total_transactions = len(all_transactions)
    recurring_transaction_count = sum(p['occurrences'] for p in recurring_payments)

    col1, col2, col3 = st.columns(3)

    with col1:
        recurring_percentage = (recurring_transaction_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("Recurring Transaction %", f"{recurring_percentage:.1f}%")

    with col2:
        avg_recurring_amount = total_recurring_amount / len(recurring_payments) if recurring_payments else 0
        st.metric("Avg Recurring Amount", f"€{avg_recurring_amount:.2f}")

    with col3:
        monthly_recurring = sum(p['average_amount'] for p in recurring_payments)
        st.metric("Monthly Recurring Total", f"€{monthly_recurring:.2f}")

    # Pattern analysis
    pattern_types = {}
    for payment in recurring_payments:
        pattern = payment.get('pattern_type', 'Unknown')
        pattern_types[pattern] = pattern_types.get(pattern, 0) + 1

    if pattern_types:
        st.write("**Pattern Distribution:**")
        for pattern, count in pattern_types.items():
            st.write(f"• {pattern}: {count} billers")

def display_error_message(message: str, error_details: str = None):
    """
    Display an error message with optional details.

    Args:
        message: Main error message
        error_details: Additional error details (optional)
    """
    st.error(f"❌ {message}")
    if error_details:
        with st.expander("Error Details"):
            st.code(error_details)

def display_success_message(message: str):
    """
    Display a success message.

    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")

def display_info_message(message: str):
    """
    Display an info message.

    Args:
        message: Info message to display
    """
    st.info(f"ℹ️ {message}")

def display_warning_message(message: str):
    """
    Display a warning message.

    Args:
        message: Warning message to display
    """
    st.warning(f"⚠️ {message}")

def display_deal_comparison_results(deal_analysis: Dict):
    """
    Display detailed deal comparison results.

    Args:
        deal_analysis: Deal comparison analysis results
    """
    if not deal_analysis or not deal_analysis.get('deal_comparisons'):
        st.info("No better deals found for your current subscriptions.")
        return

    st.header("🔍 Better Deal Analysis")
    # Summary right under header: potential monthly & annual savings
    colA, colB = st.columns(2)
    with colA:
        st.metric("Potential Monthly Savings", f"€{deal_analysis['total_monthly_savings']:.2f}")
    with colB:
        st.metric("Annual Savings", f"€{deal_analysis['total_annual_savings']:.2f}")

    # Individual deal comparisons
    st.subheader("💡 Recommended Switches")

    for i, deal in enumerate(deal_analysis['deal_comparisons'], 1):
        with st.expander(f"{i}. {deal['current_provider']} → {deal['alternative_provider']}", expanded=(i <= 3)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Current:** {deal['current_provider']}")
                st.markdown(f"**Cost:** €{deal['current_amount']:.2f}/month")
                st.markdown(f"**Category:** {deal['category']}")

            with col2:
                st.markdown(f"**Alternative:** {deal['alternative_provider']}")
                st.markdown(f"**Cost:** €{deal['alternative_amount']:.2f}/month")
                st.markdown(f"**Rating:** ⭐ {deal['rating']}/5")

            # Savings highlight
            savings_color = "🟢" if deal['savings_percentage'] >= 20 else "🟡" if deal['savings_percentage'] >= 10 else "🟠"
            st.success(f"{savings_color} **Save €{deal['monthly_savings']:.2f}/month** "
                      f"({deal['savings_percentage']:.1f}% savings)")

            st.markdown("---")

    # Action items
    st.subheader("📋 Next Steps")
    st.markdown("""
    1. **Review alternatives** - Check ratings and terms for each provider
    2. **Compare features** - Ensure the alternative meets your needs
    3. **Check contracts** - Review cancellation terms for current providers
    4. **Switch gradually** - Consider switching one service at a time
    """)
