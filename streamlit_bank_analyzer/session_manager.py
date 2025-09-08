"""
Session manager for handling Streamlit app state and data persistence.
"""

import streamlit as st
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

class SessionManager:
    """Manages session state and data persistence for the Streamlit app."""

    def __init__(self):
        """Initialize the session manager."""
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables if they don't exist."""
        if 'session_data' not in st.session_state:
            st.session_state.session_data = {
                'user_name': '',
                'image_paths': [],
                'results': [],
                'analysis_results': None,
                'session_id': None
            }

    def save_image_paths(self, user_name: str, image_paths: List[str]):
        """
        Save generated image paths to session state (without OCR results).

        Args:
            user_name: Name entered by the user
            image_paths: List of generated image file paths
        """
        st.session_state.session_data.update({
            'user_name': user_name,
            'image_paths': image_paths,
            'results': [],  # Clear any previous results
            'analysis_results': None
        })

    def save_results(self, user_name: str, image_paths: List[str], results: List[Dict]):
        """
        Save processing results to session state.

        Args:
            user_name: Name entered by the user
            image_paths: List of generated image file paths
            results: OCR processing results
        """
        st.session_state.session_data.update({
            'user_name': user_name,
            'image_paths': image_paths,
            'results': results,
            'analysis_results': None  # Reset analysis when new data is loaded
        })

    def save_analysis_results(self, analysis_results: Dict):
        """
        Save analysis results to session state.

        Args:
            analysis_results: Results from recurring payment analysis
        """
        st.session_state.session_data['analysis_results'] = analysis_results

    def get_results(self) -> Optional[Dict]:
        """
        Get current session results.

        Returns:
            Dictionary containing session data or None if no data
        """
        data = st.session_state.session_data
        if data['image_paths'] and data['results']:
            return {
                'user_name': data['user_name'],
                'image_paths': data['image_paths'],
                'results': data['results'],
                'analysis_results': data.get('analysis_results')
            }
        return None

    def get_analysis_results(self) -> Optional[Dict]:
        """
        Get analysis results if available.

        Returns:
            Analysis results dictionary or None
        """
        return st.session_state.session_data.get('analysis_results')

    def has_results(self) -> bool:
        """
        Check if session has processing results.

        Returns:
            True if results exist, False otherwise
        """
        data = st.session_state.session_data
        return bool(data['image_paths'] and data['results'])

    def has_analysis(self) -> bool:
        """
        Check if session has analysis results.

        Returns:
            True if analysis results exist, False otherwise
        """
        return st.session_state.session_data.get('analysis_results') is not None

    def clear_session(self):
        """Clear all session data."""
        st.session_state.session_data = {
            'user_name': '',
            'image_paths': [],
            'results': [],
            'analysis_results': None,
            'session_id': None
        }

    def get_user_name(self) -> str:
        """
        Get the current user name.

        Returns:
            User name string
        """
        return st.session_state.session_data.get('user_name', '')

    def get_image_paths(self) -> List[str]:
        """
        Get the current image paths.

        Returns:
            List of image file paths
        """
        return st.session_state.session_data.get('image_paths', [])

    def get_processing_results(self) -> List[Dict]:
        """
        Get the OCR processing results.

        Returns:
            List of processing result dictionaries
        """
        return st.session_state.session_data.get('results', [])

    def export_session_data(self, export_path: str = "session_export.json"):
        """
        Export current session data to a JSON file.

        Args:
            export_path: Path for the export file
        """
        try:
            data = st.session_state.session_data.copy()

            # Convert any non-serializable objects if needed
            # (Currently all data should be JSON serializable)

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error exporting session data: {e}")
            return False

    def import_session_data(self, import_path: str = "session_export.json"):
        """
        Import session data from a JSON file.

        Args:
            import_path: Path to the import file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(import_path):
                return False

            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate data structure
            required_keys = ['user_name', 'image_paths', 'results']
            if not all(key in data for key in required_keys):
                return False

            st.session_state.session_data = data
            return True
        except Exception as e:
            print(f"Error importing session data: {e}")
            return False
