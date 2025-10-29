import streamlit as st
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.ui_steps import render_steps  # keep imported; optional
from assets.ui_components import apply_theme, divider, page_header, section_header

st.set_page_config(
    page_title="Moodle Log Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# Initialize session state managers ONCE
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

apply_theme()

page_header(
    "Moodle Log Analyzer",
    "Transform your Moodle discussion data into actionable insights.", icon="🎓",
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Centered button inside the middle column
    if st.button(
        "🚀 Upload your data",
        key="pulse",
        type="primary",
        help="Go to the Data Upload page",
        use_container_width=True,  # makes it stretch nicely in center
    ):
        st.switch_page("pages/1_📊_Data_Upload.py")

section_header("Capabilities")
st.markdown(
    """
    <div class="grid-12">
      <div class="card span-4">
        <div class="panel__icon">📈</div>
        <h3>Analyze Student Activity</h3>
        <p>Understand posting rhythm, streaks, and participation trends without wading through raw tables.</p>
      </div>
      <div class="card span-4">
        <div class="panel__icon">📊</div>
        <h3>Visual Reports</h3>
        <p>Interactive plots and compact summaries made for review meetings and quick exports.</p>
      </div>
      <div class="card span-4">
        <div class="panel__icon">🎯</div>
        <h3>Insights</h3>
        <p>Surface risk, diligence, and understanding signals early enough for real interventions.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

divider()

section_header("At a glance")
st.markdown(
    """
    <div class="info-band">
      <div class="info-item"><strong>📁 Supported Formats</strong><br><span>CSV and Excel logs</span></div>
      <div class="info-item"><strong>🔒 Secure Processing</strong><br><span>All data stays in-session</span></div>
      <div class="info-item"><strong>⚡ Fast Analysis</strong><br><span>Pipeline optimised for cohorts</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

