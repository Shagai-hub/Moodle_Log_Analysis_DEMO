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

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
        height: 220px;
        display: flex;
        flex-direction: column;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .feature-card h3 {
        color: #2c3e50;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    .feature-card p {
        color: #555;
        font-size: 1rem;
        line-height: 1.5;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .cta-section {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        margin-top: 2rem;
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Mobile Responsive Styles */
    @media only screen and (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
        .main-header p {
            font-size: 1rem !important;
        }
        .main-header {
            padding: 1.5rem 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        .feature-card {
            height: auto !important;
            min-height: 180px;
            margin-bottom: 1rem !important;
            padding: 1rem !important;
        }
        .feature-card h3 {
            font-size: 1.1rem !important;
        }
        .feature-card p {
            font-size: 0.9rem !important;
        }
        .feature-icon {
            font-size: 2rem !important;
        }
        .stButton > button {
            padding: 0.6rem 1rem !important;
            font-size: 1rem !important;
        }
        [data-testid="column"] {
            padding: 0.25rem !important;
        }
    }
    
    @media only screen and (max-width: 480px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        .main-header p {
            font-size: 0.9rem !important;
        }
        .feature-card {
            min-height: 150px;
        }
        .feature-card h3 {
            font-size: 1rem !important;
        }
        .feature-card p {
            font-size: 0.85rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1>🎓 Moodle Log Analyzer</h1>
        <p>Transform your Moodle data into actionable insights</p>
    </div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("### ✨ What You Can Do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3>Analyze Student Activity</h3>
            <p>Deep dive into student engagement patterns and learning behaviors with comprehensive analytics.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Visual Reports</h3>
            <p>Generate beautiful, interactive visualizations that make data interpretation effortless.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Actionable Insights</h3>
            <p>Identify trends and patterns to improve course design and student outcomes.</p>
        </div>
    """, unsafe_allow_html=True)

# Getting Started Section
st.markdown("### 🚀 Getting Started")
st.info("""
    **Ready to begin?** Upload your Moodle log data to start analyzing student activity, 
    course engagement, and learning patterns. The process is simple and takes just a few clicks!
""")

# Call to Action
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.button("📤 Upload Your Data Now", key="goto_upload", help="Navigate to the data upload page"):
        st.switch_page("pages/1_📊_Data_Upload.py")

# Footer Info
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**📁 Supported Formats**")
    st.caption("CSV, Excel, and other log formats")

with col_b:
    st.markdown("**🔒 Secure Processing**")
    st.caption("Your data stays private and secure")

with col_c:
    st.markdown("**⚡ Fast Analysis**")
    st.caption("Get insights in seconds")