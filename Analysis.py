import streamlit as st
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.ui_steps import render_steps  # keep imported; optional
import pathlib

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

def load_css(file_path: pathlib.Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# External CSS first (lets users override defaults if they want)
load_css(pathlib.Path("assets/styles.css"))

# Baseline design system (dark-first, accessible)
st.markdown("""
<style>
:root{
  --bg: #0b1220;
  --panel: #0f172a;
  --card: #121a2c;
  --muted: #9aa3b2;
  --text: #e6e9ef;
  --text-dim: #cdd3dd;
  --accent: #7c3aed;    /* primary */
  --accent-2: #06b6d4;  /* secondary */
  --ring: rgba(124,58,237,0.4);
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
  --radius: 14px;
  --radius-sm: 10px;
  --gap: 16px;
}

@media (prefers-color-scheme: light) {
  :root{
    --bg:#f7f9fc; --panel:#ffffff; --card:#ffffff;
    --text:#0b1220; --text-dim:#324055; --muted:#6b7280;
    --shadow: 0 8px 20px rgba(18,27,40,0.08);
    --ring: rgba(124,58,237,0.25);
  }
}

/* Page background */
html, body, [class*="stApp"] {
  background:
    radial-gradient(900px 600px at 10% 10%, rgba(124,58,237,0.10), transparent 60%),
    radial-gradient(800px 500px at 90% 90%, rgba(6,182,212,0.08), transparent 60%),
    var(--bg) !important;
  color: var(--text);
}

/* Layout paddings tightened for wide layouts */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

/* HERO */
.main-header {
  text-align:center;
  padding: 0 0 0.5rem 0;
  color: var(--text);
  margin-top: 3rem;
}
.main-header h1{
  font-size: clamp(2.4rem, 6vw, 4.2rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  margin: 0px;
}
.main-header p{
  margin-top: .4rem;
  color: var(--text-dim);
  font-size: clamp(1rem, 1.6vw, 1.15rem);
}

/* Section headings */
.section-title{
  display:flex; align-items:center; gap:.6rem;
  font-weight: 800; letter-spacing:-0.02em;
  margin: 1.2rem 0 .8rem 0;
  color: var(--text);
  margin-top: 5rem;
}
.section-title .dot{
  width:10px; height:10px; border-radius:50%;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 0 0 4px rgba(124,58,237,0.12);
}

/* Panels */
.panel {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--panel);
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: var(--radius);
  padding: 1.25rem;
  box-shadow: var(--shadow);
}

/* Feature grid */
.features{
  display:grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--gap);
}
.feature-card{
  grid-column: span 4;
  background: var(--card);
  border:1px solid rgba(148,163,184,0.12);
  border-radius: var(--radius);
  padding: 1.1rem;
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  min-height: 160px;
}
.feature-card:hover{
  transform: translateY(-2px);
  border-color: rgba(124,58,237,0.35);
  box-shadow: 0 14px 40px rgba(124,58,237,0.15);
}
.feature-card h3{
  margin: 0 0 .35rem 0; font-size: 1.05rem; color: var(--text);
}
.feature-card p{
  margin:0; color: var(--text-dim); line-height: 1.55;
  font-size: .98rem;
}
.feature-icon{
  display:inline-grid; place-items:center;
  width:40px; height:40px; border-radius:10px;
  background: linear-gradient(135deg, rgba(124,58,237,0.20), rgba(6,182,212,0.18));
  border:1px solid rgba(148,163,184,0.16);
  margin-bottom:.6rem; font-size: 1.1rem;
}

/* CTA block */
.cta {
  margin-top: 1.2rem;
  padding: 1.25rem;
  border-radius: var(--radius);
  background:
    radial-gradient(600px 200px at 20% 20%, rgba(124,58,237,0.20), transparent 60%),
    radial-gradient(600px 200px at 80% 40%, rgba(6,182,212,0.20), transparent 60%),
    var(--panel);
  border:1px solid rgba(148,163,184,0.12);
  box-shadow: var(--shadow);
  text-align:center;
}
.cta p{
  margin:.2rem 0 1rem 0; color: var(--text-dim);
}

/* Streamlit button reset and CTA emphasis */
.stButton > button {
  width: min(560px, 92%);
  margin: 0 auto;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: #fff !important;
  border: none !important;
  padding: 0.95rem 1.4rem !important;
  font-size: 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: .3px !important;
  border-radius: 999px !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.35);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button:hover { transform: translateY(-1px) scale(1.01); filter: brightness(1.05); }
.stButton > button:active { transform: translateY(0) scale(.995); }

/* Info band */
.kv{
  display:grid; grid-template-columns: repeat(12, 1fr); gap: var(--gap);
}
.kv-item{
  grid-column: span 4;
  background: var(--card);
  border:1px solid rgba(148,163,184,0.12);
  border-radius: var(--radius-sm);
  padding: .9rem 1rem;
}
.kv-item strong{ color: var(--text); }
.kv-item span{ color: var(--muted); }

/* Divider */
hr{ border: none; border-top:1px solid rgba(148,163,184,0.15); margin: 1.2rem 0; }

/* Mobile */
@media (max-width: 1100px){
  .feature-card{ grid-column: span 6; }
  .kv-item{ grid-column: span 6; }
}
@media (max-width: 680px){
  .feature-card{ grid-column: span 12; }
  .kv-item{ grid-column: span 12; }
  .block-container { padding-left: .6rem; padding-right: .6rem; }
}

/* Reduced motion: chill the animations */
@media (prefers-reduced-motion: reduce){
  *{ transition: none !important; animation: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="main-header">
  <h1>Moodle Log Analyzer</h1>
  <p>Transform your Moodle data into actionable insights <br> without wrestling with spreadsheets.</p>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Upload your data", key="pulse", type="primary", help="Go to the Data Upload page", width="stretch"):
    st.switch_page("pages/1_📊_Data_Upload.py")


# ---------- FEATURES ----------
st.markdown('<div class="section-title"><span class="dot"></span><span>Capabilities</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="features">
  <div class="feature-card">
    <div class="feature-icon">📈</div>
    <h3>Analyze Student Activity</h3>
    <p>Understand patterns across sessions, posts, deadlines and streaks. No guesswork, just signals.</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <h3>Visual Reports</h3>
    <p>Interactive plots and compact tables that don’t fry your eyes. Export when you actually need a PDF.</p>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <h3>Insights</h3>
    <p>Surface risk, diligence and understanding indicators so instructors can act before week 12 chaos.</p>
  </div>
</div>
""", unsafe_allow_html=True)




# ---------- INFO BAND ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="dot"></span><span>At a glance</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="kv">
  <div class="kv-item"><strong>📁 Supported Formats</strong><br><span>CSV and Excel logs</span></div>
  <div class="kv-item"><strong>🔒 Secure Processing</strong><br><span>In-session only</span></div>
  <div class="kv-item"><strong>⚡ Fast Analysis</strong><br><span>Optimized pipeline</span></div>
</div>
""", unsafe_allow_html=True)

