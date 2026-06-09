import streamlit as st
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.ui_steps import render_steps  # keep imported; optional
from assets.ui_components import apply_theme, divider, page_header, section_header

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"

st.set_page_config(
    page_title="Moodle Log Analyzer",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state,
)


# Initialize session state managers 
if "data_manager" not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if "config" not in st.session_state:
    st.session_state.config = ConfigManager()

apply_theme()

page_header(
    "Moodle Log Analyzer",
    "A structured analysis workflow for Moodle discussion logs, student activity metrics, ranking, and COCO evaluation.",
)

act1, act2, act3 = st.columns([1, 2, 1])

with act2:
    # Centered button inside the middle column
    if st.button(
        "Upload Dataset",
        key="home_upload_dataset_btn",
        help="Go to the Data Upload page",
        use_container_width=True, 
    ):
        st.switch_page("pages/1_Data_Upload.py")

    if st.button(
        "Open Help Center",
        key="open_help_center_btn",
        help="Open setup and troubleshooting guidance.",
        use_container_width=True,
    ):
        st.switch_page("pages/9_Help.py")

section_header("Capabilities")
st.markdown(
    """
    <div class="grid-12">
      <div class="card span-4">
        <h3>Student Activity Analysis</h3>
        <p>Compute structured attributes from discussion participation.</p>
      </div>
      <div class="card span-4">
        <h3>Ranking and Evaluation</h3>
        <p>Prepare ranked matrices and run COCO-based evaluation.</p>
      </div>
      <div class="card span-4">
        <h3>Cohort Interpretation</h3>
        <p>Review watchlists, validation outputs, and presentation-ready summaries.</p>
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
      <div class="info-item"><strong>Supported Formats</strong><br><span>CSV and Excel logs</span></div>
      <div class="info-item"><strong>Session-Based Processing</strong><br><span>Uploaded data is managed within the active session</span></div>
      <div class="info-item"><strong>Reproducible Workflow</strong><br><span>Export configuration, attribute matrices, ranking, and COCO results</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
