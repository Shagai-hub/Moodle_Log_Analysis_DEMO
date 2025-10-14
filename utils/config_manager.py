import streamlit as st
import pandas as pd

class ConfigManager:
    def __init__(self):
        self.professors = ["professor_1", "professor_2"]
        self.deadlines = {
            "Quasi Exam I": pd.to_datetime("2024-11-09 00:00:00"),
            # ... your existing deadlines
        }
    
    def render_sidebar_config(self):
        # Add UI for configuring professors, deadlines, etc.
        pass