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
from display_helpers import display_statement_with_extractions, display_analysis_results

# Import the existing simple_image_reader
import sys
import os
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import the real SimpleImageReader now that imports are fixed
from simple_image_reader.simple_image_reader import SimpleImageReader

# Configure page
st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session manager
session_manager = SessionManager()

def main():
    """Main application function."""

    # Header
    st.title("🏦 Bank Statement Analyzer")
    st.markdown("""
    Generate bank statements, extract transaction data, and identify recurring payments.
    """)

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
            generate_button = st.button("🚀 Generate Statements", type="primary")
        with col2:
            ocr_button = st.button("🔍 Process with OCR", disabled=not session_manager.get_image_paths())

    # Main content area
    if generate_button and user_name.strip():
        with st.spinner("Generating bank statements..."):
            success = generate_statements_only(user_name.strip(), num_statements)

            if success:
                st.success(f"✅ Successfully generated {num_statements} statements!")
                st.rerun()  # Refresh to enable OCR button
            else:
                st.error("❌ Failed to generate statements. Please try again.")

    if ocr_button:
        with st.spinner("Processing images with OCR..."):
            success = process_ocr_only()

            if success:
                st.success("✅ Successfully processed images with OCR!")
                st.rerun()  # Refresh to show results
            else:
                st.error("❌ Failed to process images with OCR. Please try again.")

    # Display images if available
    if session_manager.get_image_paths():
        display_images_section()

    # Display results if available
    if session_manager.has_results():
        display_results_section()

        # Enable analysis button if we have data
        if st.sidebar.button("📊 Analyze Recurring Payments", type="secondary"):
            with st.spinner("Analyzing recurring payments..."):
                analyze_payments()

def generate_statements_only(name: str, num_statements: int) -> bool:
    """Generate bank statements without OCR processing."""
    try:
        # Generate statements
        st.info("📝 Generating bank statements...")
        progress_bar = st.progress(0)

        image_paths = generate_statements_for_name(name, num_statements)

        if not image_paths:
            st.error("Failed to generate statements")
            return False

        progress_bar.progress(100)
        progress_bar.empty()

        # Save image paths to session (without OCR results)
        session_manager.save_image_paths(name, image_paths)

        return True

    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        return False

def process_ocr_only() -> bool:
    """Process existing images with OCR."""
    try:
        # Get current image paths
        image_paths = session_manager.get_image_paths()

        if not image_paths:
            st.error("No images available for OCR processing")
            return False

        # Process with OCR
        st.info("🔍 Processing images with OCR...")
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
        st.error(f"Error during OCR processing: {str(e)}")
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
        with st.expander(f"📄 Statement {i+1}: {Path(image_path).name}", expanded=(i < 3)):
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

if __name__ == "__main__":
    main()
