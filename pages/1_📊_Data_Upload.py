import streamlit as st
import pandas as pd
import os
import pathlib
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.ui_steps import render_steps

# Safe init if this page is opened directly
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

render_steps(active="1 Analyze")
def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config

    # ---------- DESIGN SYSTEM (consistent with Home) ----------
    st.markdown("""
    <style>
    :root{
      --bg: #000B18;
      --panel: #0f172a;
      --card: #121a2c;
      --muted: #9aa3b2;
      --text: #e6e9ef;
      --text-dim: #cdd3dd;
      --accent: #7c3aed;
      --accent-2: #06b6d4;
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

    html, body, [class*="stApp"] {
      background:
        radial-gradient(900px 600px at 10% 10%, rgba(124,58,237,0.10), transparent 60%),
        radial-gradient(800px 500px at 90% 90%, rgba(6,182,212,0.08), transparent 60%),
        var(--bg) !important;
      color: var(--text);
    }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

    /* Section heading */
    .section-title{
      display:flex; align-items:center; gap:.6rem;
      font-weight: 800; letter-spacing:-0.02em;
      margin: 0 0 .8rem 0;
      color: var(--text);
    }
    .section-title .dot{
      width:10px; height:10px; border-radius:50%;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 0 0 4px rgba(124,58,237,0.12);
    }

    /* Panels and cards */
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--panel);
      border: 1px solid rgba(148,163,184,0.12);
      border-radius: var(--radius);
      padding: 1.2rem;
      box-shadow: var(--shadow);
    }
    .card {
      background: var(--card);
      border: 1px solid rgba(148,163,184,0.12);
      border-radius: var(--radius);
      padding: 1rem 1.1rem;
      box-shadow: var(--shadow);
    }

    /* Grid */
    .grid-12 { display:grid; grid-template-columns: repeat(12, 1fr); gap: var(--gap); }
    .span-12{ grid-column: span 12; }
    .span-6{ grid-column: span 6; }
    .span-4{ grid-column: span 4; }

    @media (max-width: 1100px){ .span-6{ grid-column: span 12; } .span-4{ grid-column: span 6; } }
    @media (max-width: 680px){ .span-4{ grid-column: span 12; } .block-container{ padding-left:.6rem; padding-right:.6rem; } }

    /* Header (page-local) */
    .upload-hero{
      text-align:center;
      padding: .6rem 0 1.1rem 0;
      margin-top: 24px;
    }
    .upload-hero h1{
      margin:.1rem 0 .2rem 0;
      font-size: clamp(1.9rem, 4.5vw, 2.6rem);
      font-weight: 800; letter-spacing:-0.03em;
    }
    .upload-hero p{
      color: var(--text-dim);
      font-size: clamp(.98rem, 1.6vw, 1.05rem);
      margin:0;
    }

    /* Buttons */
    .stButton > button {
      background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
      color: #fff !important; border: none !important;
      padding: .9rem 1.1rem !important; font-weight: 800 !important;
      letter-spacing: .2px !important; border-radius: 999px !important;
      box-shadow: 0 10px 28px rgba(124,58,237,0.35);
      transition: transform .15s ease, filter .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover{ transform: translateY(-1px) scale(1.01); filter:brightness(1.05); }
    .stButton > button:active{ transform: translateY(0) scale(.995); }
    .stButton > button:focus-visible{
      outline:none !important;
      box-shadow: 0 0 0 3px var(--panel), 0 0 0 6px var(--ring), 0 16px 42px rgba(124,58,237,.42) !important;
    }
    
    /* Download buttons: consistent with primary buttons */
    .stDownloadButton > button {
      background: linear-gradient(135deg, var(--panel), var(--card)) !important;
      color: var(--text) !important;
      border: 1px solid rgba(148,163,184,0.18) !important;
      padding: .9rem 1.1rem !important;
      font-weight: 800 !important;
      letter-spacing: .2px !important;
      border-radius: 999px !important;
      box-shadow: 0 8px 22px rgba(2, 6, 23, 0.35);
      transition: transform .15s ease, filter .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    
    /* Hover: slight lift and brighter border */
    .stDownloadButton > button:hover{
      transform: translateY(-1px) scale(1.01);
      filter: brightness(1.05);
      border-color: rgba(124,58,237,0.35) !important;
      box-shadow: 0 12px 32px rgba(124,58,237,0.30);
    }
    
    /* Focus ring aligned with system */
    .stDownloadButton > button:focus-visible{
      outline: none !important;
      box-shadow:
        0 0 0 3px var(--panel),
        0 0 0 6px var(--ring),
        0 16px 42px rgba(124,58,237,0.42) !important;
    }
    
    /* Optional: key-specific accent if you want subtle differentiation */
    .st-key-dl_csv > button{
      background: linear-gradient(135deg, var(--card), rgba(124,58,237,0.12)) !important;
    }
    .st-key-dl_xlsx > button{
      background: linear-gradient(135deg, var(--card), rgba(6,182,212,0.12)) !important;
    }

    /* Metrics: denser and readable in dark */
    div[data-testid="stMetricValue"] > span {
      color: var(--text) !important;
    }
    div[data-testid="stMetricLabel"] > div {
      color: var(--muted) !important;
    }

    /* Expander polish */
    .streamlit-expanderHeader { font-weight: 700; color: var(--text); }
    .stExpander { border: 1px solid rgba(148,163,184,0.12); border-radius: var(--radius); }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
    </style>
    """, unsafe_allow_html=True)

    # ---------- HEADER ----------
    st.markdown("""
    <div class="upload-hero">
      <h1>📥 Upload Discussion Data</h1>
      <p>Bring in your Moodle discussion logs to kick off analysis and downstream insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # ---------- UPLOAD ----------
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xlsx"],
        help="File should contain columns like userid, userfullname, message, created, etc.",
        label_visibility="visible",
        width="stretch"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="panel">
      <strong>Supported formats:</strong> CSV, XLSX<br>
      <span style="color:var(--muted)">Expected columns: <code>userid</code>, <code>userfullname</code>, <code>message</code>, optional <code>created</code>, <code>modified</code></span>
    </div>
    """, unsafe_allow_html=True)
   # ---------- DEMO DATA ----------
    st.markdown('<div class="section-title" style="margin-top:1rem;"><span class="dot"></span><span>Download sample data for demonstartion</span></div>', unsafe_allow_html=True)
    
    col_demo_csv, col_demo_xlsx = st.columns(2)
    
    with col_demo_csv:
        sample_csv_path = os.path.join("sample_data", "discussion_demo.csv")
        try:
            with open(sample_csv_path, "rb") as sample_csv:
                st.download_button(
                    label="📥 Download Demo CSV",
                    data=sample_csv.read(),
                    file_name="discussion_demo.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv"  # used by CSS above
                )
        except FileNotFoundError:
            st.warning("Demo CSV not found at sample_data/discussion_demo.csv")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_demo_xlsx:
        sample_xlsx_path = os.path.join("sample_data", "discussion_demo.xlsx")
        try:
            with open(sample_xlsx_path, "rb") as sample_xlsx:
                st.download_button(
                    label="📥 Download Demo XLSX",
                    data=sample_xlsx.read(),
                    file_name="discussion_demo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_xlsx"  # used by CSS above
                )
        except FileNotFoundError:
            st.warning("Demo XLSX not found at sample_data/discussion_demo.xlsx")
        st.markdown('</div>', unsafe_allow_html=True)
    


    # ---------- PROCESS ----------
    if uploaded_file:
        with st.spinner("🔄 Processing your file..."):
            df = process_uploaded_file(uploaded_file)

        if df is not None:
            data_manager.store_raw_data(df, source_info=uploaded_file.name)
            st.success(f"✅ Loaded '{uploaded_file.name}'")

            # Overview
            st.markdown('<div class="section-title"><span class="dot"></span><span>Dataset overview</span></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total rows", f"{df.shape[0]:,}")
            with c2: st.metric("Total columns", df.shape[1])
            with c3:
                uniq_users = df['userfullname'].nunique() if 'userfullname' in df.columns else "—"
                st.metric("Unique users", uniq_users)

            # Preview
            st.markdown('<div class="section-title"><span class="dot"></span><span>Data preview</span></div>', unsafe_allow_html=True)
            with st.expander("First 10 rows", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Actions
          
            act1, act2 = st.columns(2)
            with act1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download processed data",
                    csv_data,
                    f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                    key="dl_processed"
                )
            with act2:
                if st.button("⚙️ Go to Configuration", use_container_width=True, key="go_to_config"):
                    try:
                        st.switch_page("pages/2_⚙️_Configuration.py")
                    except Exception:
                        st.warning("Can’t auto-navigate. Open “⚙️ Configuration” from the sidebar.")

def process_uploaded_file(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    df = None

    if file_ext == ".csv":
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1250"]
        seps = [",", ";", "\t"]
        for enc in encodings:
            for sep in seps:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                    if df is not None:
                        break
                except Exception:
                    df = None
            if df is not None:
                break

    elif file_ext == ".xlsx":
        try:
            import openpyxl  # noqa
        except ImportError:
            st.error("Missing 'openpyxl'. Install with: `pip install openpyxl`")
            return None
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Failed to read '{uploaded_file.name}': {e}")
            return None

    if df is None:
        st.error(f"❌ Failed to read '{uploaded_file.name}'. Check the format.")
        return None

    # Validate required columns
    required = ['userid', 'userfullname', 'message']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.info("Columns needed: userid, userfullname, message")
        return None

    # Parse dates if present
    for col in ['created', 'modified']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not parse '{col}' as datetime: {e}")

    # Clean column names
    df.columns = [
        (c.replace(" ", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", ""))
        for c in df.columns
    ]
    return df

if __name__ == "__main__":
    main()
