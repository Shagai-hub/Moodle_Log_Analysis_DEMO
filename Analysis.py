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

PAGE_CSS = '''
:root{
  --bg1: #0f172a; --bg2:#07103a; --glass: rgba(255,255,255,0.04);
  --accent1: #7c3aed; --accent2: #06b6d4; --muted:#9aa3b2;
}
html, body, [class*="stApp"] {
  background: radial-gradient(1000px 600px at 10% 10%, rgba(124,58,237,0.12), transparent),
              radial-gradient(800px 500px at 90% 90%, rgba(6,182,212,0.08), transparent),
              linear-gradient(180deg, var(--bg1), var(--bg2));
  color: #e6eef8;
}

.header { display:flex; align-items:center; justify-content:space-between; gap:12px;}
.logo { font-weight:700; font-size:20px; display:flex; align-items:center; gap:10px;}
.logo .dot { width:12px; height:12px; border-radius:50%; background:linear-gradient(45deg,var(--accent1),var(--accent2)); box-shadow:0 6px 24px rgba(6,182,212,0.12); }
.nav { color:var(--muted); font-size:14px; }
.hero {
  padding:28px; border-radius:16px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  box-shadow: 0 10px 40px rgba(2,6,23,0.6); margin-top:18px;
}
.title { font-size:34px; font-weight:700; margin-bottom:6px; }
.lead { color:var(--muted); margin-bottom:18px; }
.cta { display:flex; gap:12px; }
.btn { padding:10px 16px; border-radius:10px; font-weight:600; cursor:pointer; }
.btn-primary { background: linear-gradient(90deg,var(--accent1),var(--accent2)); color:white; border:none; }
.btn-ghost { background:transparent; border:1px solid rgba(255,255,255,0.06); color:var(--muted); }

.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top:18px; }
.card { padding:18px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); min-height:120px; }
.card .eyebrow { color:var(--muted); font-size:12px; margin-bottom:8px; }
.card h4 { margin:0 0 8px 0; }
.blob { width:60px; height:60px; border-radius:18px; background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,182,212,0.12)); display:inline-block; float:right; }

.preview { margin-top:18px; padding:18px; border-radius:14px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); }
.preview .mock { height:220px; border-radius:10px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: inset 0 1px 0 rgba(255,255,255,0.02); display:flex; align-items:center; justify-content:center; color:var(--muted); font-weight:600; }
.footer { margin-top:28px; color:var(--muted); font-size:13px; }

@keyframes floaty { 0%{transform:translateY(0)}50%{transform:translateY(-6px)}100%{transform:translateY(0)} }
.card .blob { animation: floaty 6s ease-in-out infinite; }

@media (max-width:900px){ .grid{ grid-template-columns: 1fr; } .title { font-size:28px; } }
'''

st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)

# Header
with st.container():
    st.markdown(
        """
        <div class='header'>
          <div class='logo'><div class='dot'></div>Moodle Visual</div>
          <div class='nav'>Home &nbsp;&nbsp; Features &nbsp;&nbsp; About</div>
        </div>
        """, unsafe_allow_html=True)

# Hero
with st.container():
    st.markdown(
        """
        <div class='hero'>
          <div class='title'>Beautifully simple overview — visuals only.</div>
          <div class='lead'>A clean, modern landing page designed to present UI concepts and brand visuals without showing any raw data or analytics. Perfect for demos, presentations, or as a wrapper around your real dashboard.</div>
          <div class='cta'>
            <button class='btn btn-primary' onclick="window.location.href='#preview'">Explore visuals</button>
            <button class='btn btn-ghost' onclick="window.location.href='#contact'">Brand settings</button>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Feature grid (visual-only cards)
with st.container():
    st.markdown("""
    <div class='grid'>
      <div class='card'>
         <div class='eyebrow'>Design system</div>
         <h4>Consistent components</h4>
         <div class='blob'></div>
         <div class='small'>Reusable cards, buttons, and tones for polished UIs.</div>
      </div>
      <div class='card'>
         <div class='eyebrow'>Branding</div>
         <h4>Custom color palettes</h4>
         <div class='blob'></div>
         <div class='small'>Swap gradients and accents to match your school or project.</div>
      </div>
      <div class='card'>
         <div class='eyebrow'>Interactions</div>
         <h4>Micro-animations</h4>
         <div class='blob'></div>
         <div class='small'>Subtle motion that makes the interface feel alive — without noise.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Mock preview area (no data shown)
with st.container():
    st.markdown("""
    <div id='preview' class='preview'>
      <div class='mock'>Preview: Visual theme + component samples (no data)</div>
    </div>
    """, unsafe_allow_html=True)

# Brand customizer (visual controls simulated)
with st.container():
    st.markdown("""
    <div style='display:flex; gap:12px; margin-top:12px;'>
      <div style='flex:1'>
        <div class='card'>
          <div class='eyebrow'>Accent</div>
          <h4>Primary & secondary</h4>
          <div style='display:flex; gap:8px; margin-top:8px;'>
            <div style='width:40px; height:40px; border-radius:8px; background:linear-gradient(45deg,#7c3aed,#06b6d4)'></div>
            <div style='width:40px; height:40px; border-radius:8px; background:#06b6d4'></div>
            <div style='width:40px; height:40px; border-radius:8px; background:#7c3aed'></div>
          </div>
          <div class='small' style='margin-top:10px;color:var(--muted)'>This section is visual only — replace with brand tokens as needed.</div>
        </div>
      </div>
      <div style='width:320px'>
        <div class='card'>
          <div class='eyebrow'>Theme</div>
          <h4>Dark by default</h4>
          <div class='small' style='color:var(--muted)'>Designed for presentation screens and low-light rooms.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
with st.container():
    st.markdown("""
    <div class='footer'>
      Built as a visuals-only landing page. To integrate real components, inject your SessionDataManager into <code>st.session_state['data_manager']</code> and swap the preview for your dashboard component. No automatic loading will occur here — we obey your request: visuals only.
    </div>
    """, unsafe_allow_html=True)

# Small helpful note for the developer
st.markdown("""
---
**Developer note:** This file intentionally avoids pandas/plotly and any data-handling. To re-enable data-driven components, import your data and mount them inside the preview container.
""")

if st.button("📊 Data Upload", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📊_Data_Upload.py")