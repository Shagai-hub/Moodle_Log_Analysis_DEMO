#session_data_manager.py - manage session state for data storage and retrieval
import pandas as pd
import streamlit as st
from datetime import datetime

class SessionDataManager:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all data storage"""
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'student_attributes' not in st.session_state:
            st.session_state.student_attributes = None
        if 'ranked_results' not in st.session_state:
            st.session_state.ranked_results = None
        if 'coco_results' not in st.session_state:
            st.session_state.coco_results = {}
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def store_raw_data(self, df, source_info=None):
        """Store raw data in session state"""
        st.session_state.raw_data = {
            'dataframe': df,
            'upload_time': pd.Timestamp.now(),
            'source': source_info,
            'shape': df.shape,
            'columns': list(df.columns)
        }
        self._add_to_history("Raw data uploaded", f"Shape: {df.shape}")
    
    def get_raw_data(self):
        """Get raw data DataFrame from session state"""
        if st.session_state.raw_data and 'dataframe' in st.session_state.raw_data:
            return st.session_state.raw_data['dataframe']
        return None
    
    def get_raw_data_info(self):
        """Get raw data metadata"""
        return st.session_state.raw_data
    
    def get_analysis_history(self):
        """Get the analysis history"""
        return st.session_state.analysis_history
    
    def _add_to_history(self, action, details):
        """Track analysis steps"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        st.session_state.analysis_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': action,
            'details': str(details)
        })
    
    # Add methods for other data types you'll need
    def store_student_attributes(self, df):
        st.session_state.student_attributes = df
        self._add_to_history("Student attributes computed", f"Shape: {df.shape}")
    
    def get_student_attributes(self):
        return st.session_state.student_attributes
    
    def store_ranked_results(self, df):
        st.session_state.ranked_results = df
        self._add_to_history("Ranking completed", f"Shape: {df.shape}")
    
    def get_ranked_results(self):
        return st.session_state.ranked_results
    
    def store_coco_results(self, tables):
        """Store COCO analysis results"""
        st.session_state.coco_results = tables
        self._add_to_history("COCO analysis completed", f"{len(tables)} tables generated")
    
    def get_coco_results(self, table_name=None):
        """Get COCO analysis results"""
        if table_name:
            return st.session_state.coco_results.get(table_name)
        return st.session_state.coco_results
    
    def store_validation_results(self, df):
        """Store validation results"""
        st.session_state.validation_results = df
        self._add_to_history("Validation completed", f"Results for {len(df)} students")
    
    def get_validation_results(self):
        """Get validation results"""
        return st.session_state.validation_results
    
    def clear_session(self):
        """Clear all session data"""
        self.init_session_state()  # Reset to initial state