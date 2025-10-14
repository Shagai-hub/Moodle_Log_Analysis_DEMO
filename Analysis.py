import streamlit as st
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager

# Initialize session state managers ONCE
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

st.set_page_config(
    page_title="Moodle Log Analyzer",
    page_icon="📊", 
    layout="wide"
)

st.title("🎓 Moodle Log Analyzer DEMO")
st.markdown("Navigate using the sidebar to access different analysis features.")