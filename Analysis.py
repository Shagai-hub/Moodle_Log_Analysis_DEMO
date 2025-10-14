import streamlit as st
from utils.config_manager import ConfigManager
from utils.session_data_manager import SessionDataManager

# Initialize core components
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()

st.set_page_config(
    page_title="Moodle Log Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 Moodle Log Analyzer DEMO")
st.markdown("Navigate using the sidebar to access different analysis features.")