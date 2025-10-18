
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
if st.button("📤 Go to Data Upload", key="goto_upload", help="Navigate to the data upload page"):
    st.switch_page("pages/1_📊_Data_Upload.py")