import streamlit as st
import pandas as pd
import os
import pathlib
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.ui_steps import render_steps
from assets.ui_components import apply_theme, divider, info_panel, page_header, section_header

# Safe init if this page is opened directly
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

render_steps(active="1 Analyze")
apply_theme()


def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config

    page_header(
        "Upload Discussion Data",
        "Bring in your Moodle discussion logs to kick off analysis and downstream insights.",
        icon="📥",
    )

    # ---------- UPLOAD ----------
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xlsx"],
        help="File should contain columns like userid, userfullname, message, created, etc.",
        label_visibility="visible",
        width="stretch"
    )

    divider()

    info_panel(
        "<strong>Supported formats:</strong> CSV, XLSX<br>"
        "<span style='color:var(--muted)'>Expected columns: <code>userid</code>, <code>userfullname</code>, "
        "<code>message</code>, optional <code>created</code>, <code>modified</code></span>",
        icon="🗂️",
    )

    # ---------- DEMO DATA ----------
    section_header("Download sample data for demonstration", tight=True)

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
    
    # ---------- PROCESS ----------
    if uploaded_file:
        with st.spinner("🔄 Processing your file..."):
            df = process_uploaded_file(uploaded_file)

        if df is not None:
            data_manager.store_raw_data(df, source_info=uploaded_file.name)
            st.success(f"✅ Loaded '{uploaded_file.name}'")

            # Overview
            section_header("Dataset overview", tight=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total rows", f"{df.shape[0]:,}")
            with c2: st.metric("Total columns", df.shape[1])
            with c3:
                uniq_users = df['userfullname'].nunique() if 'userfullname' in df.columns else "—"
                st.metric("Unique users", uniq_users)

            # Preview
            section_header("Data preview", tight=True)
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
                if st.button(
                    "⚙️ Go to Configuration",
                    use_container_width=True,
                    key="pulse",
                    type="primary",
                ):
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
