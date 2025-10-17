import streamlit as st

st.set_page_config(page_title="Moodle Visual - Upload", page_icon="🎨", layout='wide')

# Minimal visuals-only landing page with a single navigation button to the data upload page.
PAGE_CSS = '''
:root{
  --bg1: #0f172a; --bg2:#07103a; --accent1: #7c3aed; --accent2: #06b6d4; --muted:#9aa3b2;
}
html, body, [class*="stApp"] {
  background: radial-gradient(1000px 600px at 10% 10%, rgba(124,58,237,0.12), transparent),
              radial-gradient(800px 500px at 90% 90%, rgba(6,182,212,0.08), transparent),
              linear-gradient(180deg, var(--bg1), var(--bg2));
  color: #e6eef8;
}
.container { padding: 40px 24px; }
.card { max-width:920px; margin:auto; padding:38px; border-radius:16px; text-align:center; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 18px 60px rgba(2,6,23,0.6); }
.logo { display:flex; align-items:center; justify-content:center; gap:12px; margin-bottom:8px; }
.logo .dot { width:14px; height:14px; border-radius:50%; background:linear-gradient(45deg,var(--accent1),var(--accent2)); box-shadow:0 8px 32px rgba(6,182,212,0.12); }
.title { font-size:30px; font-weight:700; margin-bottom:6px; }
.lead { color:var(--muted); margin-bottom:22px; }
.big-btn { padding:14px 26px; border-radius:12px; font-weight:700; font-size:16px; cursor:pointer; }
.btn-primary { background: linear-gradient(90deg,var(--accent1),var(--accent2)); color:white; border:none; }
.footer { margin-top:18px; color:var(--muted); font-size:13px; }
@media (max-width:900px){ .card{ padding:22px; } .title{ font-size:22px; } }
'''

st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class='card'>
      <div class='logo'><div class='dot'></div><div style='font-weight:700'>Moodle Visual</div></div>
      <div class='title'>Ready to upload your Moodle logs</div>
      <div class='lead'>This page is purely visual. No data will be shown here. Proceed to the Data Upload page to import logs and run analyses.</div>
    </div>
    """, unsafe_allow_html=True)

# Centered button — the only interactive element on this page.
cols = st.columns([1, 2, 1])
with cols[1]:
    if st.button("📤 Go to Data Upload", key="goto_upload", help="Navigate to the data upload page"):
        # Update this target if your pages folder or filename differs.
        st.switch_page("pages/1_📊_Data_Upload.py")

st.markdown("""
<div style='text-align:center; margin-top:18px; color:#9aa3b2; font-size:13px;'>
  Built as a minimal visuals-only landing page. To change the target page, edit the string passed to <code>st.switch_page()</code>.
</div>
""", unsafe_allow_html=True)
