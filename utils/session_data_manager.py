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
        # Add more as needed...
    
    def store_raw_data(self, df, source_info):
        st.session_state.raw_data = df
        # Add to analysis history
    
    def get_raw_data(self):
        return st.session_state.raw_data
    
    # Add similar methods for other data types