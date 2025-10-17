import streamlit as st
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
import requests
from streamlit_lottie import st_lottie  # Ensure installed: pip install streamlit-lottie

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

# Inject AOS for animate on scroll
st.markdown("""
<link rel="stylesheet" href="https://unpkg.com/aos@next/dist/aos.css" />
""", unsafe_allow_html=True)

# Custom CSS for parallax and styles
st.markdown("""
<style>
    .parallax {
        background-image: url("https://example.com/education-background.jpg");  /* Replace with your image URL */
        min-height: 500px;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-align: center;
    }
    .hero-text {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    @media only screen and (max-device-width: 1366px) {
        .parallax {
            background-attachment: scroll;
        }
    }
</style>
""", unsafe_allow_html=True)

# Hero section with parallax
st.markdown("""
<div class="parallax">
    <div class="hero-text" data-aos="fade-down">
        <h1>🎓 Moodle Log Analyzer DEMO</h1>
        <p>Unlock insights into student performances with interactive analytics.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load and display Lottie animation (education-themed)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets5.lottiefiles.com/packages/lf20_7pAYLK.json"  # Example: Education analytics animation
lottie_json = load_lottieurl(lottie_url)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st_lottie(lottie_json, height=300, key="education_anim")

# Features section with scroll animations
st.markdown('<h2 data-aos="fade-up">Key Features</h2>', unsafe_allow_html=True)
cols = st.columns(3)
with cols[0]:
    st.markdown('<div data-aos="fade-right"><h3>Engagement Tracking</h3><p>Visualize student interactions.</p></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown('<div data-aos="fade-up"><h3>Performance Dashboards</h3><p>Interactive charts on demand.</p></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown('<div data-aos="fade-left"><h3>Predictive Insights</h3><p>Forecast trends with ease.</p></div>', unsafe_allow_html=True)

st.markdown("Navigate using the sidebar to access different analysis features.")


# Initialize AOS
st.markdown("""
<script src="https://unpkg.com/aos@next/dist/aos.js"></script>
<script>
    AOS.init({
        duration: 1200,
    });
</script>
""", unsafe_allow_html=True)