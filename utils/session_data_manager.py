import pandas as pd
import streamlit as st
from datetime import datetime

class SessionDataManager:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        # Initialize all data storage
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'student_attributes' not in st.session_state:
            st.session_state.student_attributes = None
    
   # Add to your existing SessionDataManager class

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

def _add_to_history(self, action, details):
    """Track analysis steps"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    st.session_state.analysis_history.append({
        'timestamp': pd.Timestamp.now(),
        'action': action,
        'details': str(details)
    })
    
    # Add similar methods for other data types