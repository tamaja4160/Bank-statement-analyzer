"""
Streamlit Bank Statement Analyzer
A comprehensive application for generating, processing, and analyzing bank statements
to identify recurring payments.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import time
import os
from typing import List, Dict, Optional
import json

# Import our custom modules
from statement_generator import generate_statements_for_name
from payment_analyzer import analyze_recurring_payments
from session_manager import SessionManager
from display_helpers import (
    apply_global_styles,
    display_statement_with_extractions,
    display_analysis_results,
    display_deal_comparison_results,
    display_processing_error,
    display_processing_success,
)

# Import the existing simple_image_reader
import sys
import os
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import the real SimpleImageReader now that imports are fixed
from simple_image_reader.simple_image_reader import SimpleImageReader

# Configure page
st.set_page_config(
    page_title="BillWise: Recurring Payments",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session manager
session_manager = SessionManager()

def main():
    """Main application function."""

    # Header
    st.title("💡 BillWise: Recurring Payments")
    st.markdown("""
    Discover, track, and save on recurring costs.
    """)

    # Onboarding and Help
    with st.expander("🚀 Quick Start Guide & Privacy", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **How it works:**
            1. **Get Statements** - Generate sample statements
            2. **Analyze** - Find recurring payments and savings opportunities
            3. **Save Money** - Get recommendations to reduce costs (up to multiple 1000s of € per year!)
            """)
        with col2:
            st.markdown("""
            **Need help?**
            - **Bank statements not generated?** Try rerunning the generation step
            - **Missing transactions?** Check image quality and try enhanced OCR
            - **Recurring payments?** These are payments that appear regularly (e.g., monthly bills)
            """)

    # Apply larger fonts for all tables (Extracted Transactions, Recurring Payment Analysis, etc.)
    apply_global_styles(table_font_px=18)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # User input
        user_name = st.text_input(
            "Enter your name:",
            placeholder="e.g., John Doe",
            help="This will be used to personalize the generated statements"
        )

        num_statements = st.slider(
            "Number of statements to generate:",
            min_value=5,
            max_value=20,
            value=5,
            help="More statements provide better recurring payment analysis"
        )

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            generate_button = st.button(
                "🚀 Generate Statements",
                type="primary",
                disabled=not user_name.strip(),
                help="Enter your name first to generate personalized statements"
            )
        with col2:
            ocr_button = st.button("🔍 Process with OCR", disabled=not session_manager.get_image_paths())

        # Reset session button
        st.markdown("---")
        if st.button("🔄 Reset Session", type="secondary", help="Clear all data and start over"):
            session_manager.clear_session()
            # Clear session state
            st.session_state.clear()
            st.success("✅ Session reset! Start fresh with new data.")
            st.rerun()

    # Main content area
    if generate_button and user_name.strip():
        with st.spinner("Generating bank statements..."):
            success = generate_statements_only(user_name.strip(), num_statements)

            if success:
                st.success("✅ Successfully generated statements! Next: Process with OCR to extract transactions.")
                st.rerun()  # Refresh to enable OCR button
            else:
                st.error("❌ Failed to generate statements. Please try again.")

    if ocr_button:
        with st.spinner("Processing images with OCR..."):
            success = process_ocr_only()

            if success:
                st.success("✅ Successfully processed images! Next: Analyze recurring payments to find savings.")
                st.rerun()  # Refresh to show results
            else:
                st.error("❌ Failed to process images with OCR. Please try again.")

    # Display images if available
    if session_manager.get_image_paths():
        display_images_section()

    # Display results if available
    if session_manager.has_results():
        display_results_section()

        # Enable analysis button if we have data - moved to main area
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Analyze Recurring Payments", type="primary", use_container_width=True):
                # Persist analysis view across reruns
                st.session_state.show_analysis = True
                st.rerun()

        # Render analysis if user previously triggered it
        if st.session_state.get('show_analysis', False):
            analyze_payments()

def generate_statements_only(name: str, num_statements: int) -> bool:
    """Generate bank statements without OCR processing."""
    try:
        # Generate statements
        progress_bar = st.progress(0)

        image_paths = generate_statements_for_name(name, num_statements)

        if not image_paths:
            display_processing_error("statement generation")
            return False

        progress_bar.progress(100)
        progress_bar.empty()

        # Save image paths to session (without OCR results)
        session_manager.save_image_paths(name, image_paths)

        return True

    except Exception as e:
        display_processing_error("statement generation", str(e))
        return False

def process_ocr_only() -> bool:
    """Process existing images with OCR."""
    try:
        # Get current image paths
        image_paths = session_manager.get_image_paths()

        if not image_paths:
            display_processing_error("OCR processing", "No images available")
            return False

        # Process with OCR
        reader = SimpleImageReader()

        all_results = []
        progress_bar = st.progress(0)

        for i, image_path in enumerate(image_paths):
            st.write(f"Processing statement {i+1}/{len(image_paths)}...")

            result = reader.process_single_image(Path(image_path))
            all_results.append(result)

            # Update progress
            progress = (i + 1) / len(image_paths) * 100
            progress_bar.progress(int(progress))

        progress_bar.progress(100)
        progress_bar.empty()

        # Save results to session
        user_name = session_manager.get_user_name()
        session_manager.save_results(user_name, image_paths, all_results)

        return True

    except Exception as e:
        display_processing_error("OCR processing", str(e))
        return False



def display_images_section():
    """Display the generated statement images before OCR processing."""
    st.header("📄 Generated Bank Statements")

    image_paths = session_manager.get_image_paths()

    if not image_paths:
        st.warning("No images available. Please generate statements first.")
        return

    st.info("💡 Click 'Process with OCR' to extract transaction data from these images.")

    # Display images in a grid
    cols = st.columns(3)  # 3 images per row

    for i, image_path in enumerate(image_paths):
        col_idx = i % 3
        with cols[col_idx]:
            try:
                st.image(image_path, caption=f"Statement {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Could not display image: {e}")

def display_results_section():
    """Display the generated statements and their extractions."""
    st.header("📋 Generated Statements & Extractions")

    results = session_manager.get_results()

    if not results:
        st.warning("No results available. Please generate statements first.")
        return

    # Summary statistics
    total_statements = len(results['image_paths'])
    total_transactions = sum(len(result.get('transactions', [])) for result in results['results'])
    successful_extractions = sum(1 for result in results['results'] if result.get('success', False))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Statements", total_statements)
    with col2:
        st.metric("Successful Extractions", successful_extractions)
    with col3:
        st.metric("Total Transactions", total_transactions)

    # Display each statement
    for i, (image_path, result) in enumerate(zip(results['image_paths'], results['results'])):
        with st.expander(f"📄 Statement {i+1}", expanded=(i < 3)):
            display_statement_with_extractions(image_path, result)

def analyze_payments():
    """Analyze recurring payments and display recommendations."""
    st.header("🔍 Recurring Payment Analysis")

    results = session_manager.get_results()
    if not results:
        st.error("No data available for analysis")
        return

    # Collect all transactions
    all_transactions = []
    for result in results['results']:
        if result.get('success', False):
            all_transactions.extend(result.get('transactions', []))

    if not all_transactions:
        st.warning("No transactions found to analyze")
        return

    # Analyze recurring payments
    analysis_results = analyze_recurring_payments(all_transactions)

    # Display results
    display_analysis_results(analysis_results, all_transactions)

    # Add "Check for Better Deals" button if we have recurring payments
    if analysis_results.get('recurring_payments'):
        st.markdown("---")
        st.write("💡 **Tip:** Click below to see if you can save money by switching providers!")

        # Debug info
        st.write(f"Found {len(analysis_results['recurring_payments'])} recurring payments")

        if st.button("🔍 Check for Better Deals", type="primary", use_container_width=True):
            # Persist intent and re-run so comparison renders deterministically
            st.session_state.show_deal_comparison = True
            st.session_state.analysis_results = analysis_results
            st.rerun()

    # Show deal comparison if button was clicked
    if st.session_state.get('show_deal_comparison', False):
        show_deal_comparison(st.session_state.get('analysis_results', {}))
def show_deal_comparison(analysis_results: Dict):
    """Show detailed deal comparison results."""
    deal_analysis = analysis_results.get('deal_analysis', {})

    if not deal_analysis or not deal_analysis.get('deal_comparisons'):
        st.info("🔍 No better deals found for your current subscriptions. Your providers are already offering competitive rates!")
        if st.button("⬅️ Back to Analysis"):
            st.session_state.show_deal_comparison = False
            st.rerun()
        return

    # Add back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back to Analysis"):
            st.session_state.show_deal_comparison = False
            st.rerun()

    # Display detailed deal comparison
    display_deal_comparison_results(deal_analysis)

if __name__ == "__main__":
    main()
