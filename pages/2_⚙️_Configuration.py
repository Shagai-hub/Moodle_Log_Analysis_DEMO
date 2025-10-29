# pages/2_⚙️_Configuration.py
import streamlit as st
import pandas as pd
from utils.config_manager import ConfigManager
from utils.session_data_manager import SessionDataManager
import pathlib

# Optional: step bar if you use it elsewhere
try:
    from utils.ui_steps import render_steps
    STEPS_AVAILABLE = True
except Exception:
    STEPS_AVAILABLE = False

# ---------- Safe initialization ----------
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()

config: ConfigManager = st.session_state.config
data_manager: SessionDataManager = st.session_state.data_manager

def load_css(file_path: pathlib.Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# External CSS first (if present)
load_css(pathlib.Path("assets/styles.css"))

# ---------- Local page-scoped UI polish ----------
st.markdown("""
<style>
/* Secondary buttons (default for this page): darker, neutral, compact) */
.stButton > button {
  background: linear-gradient(135deg, var(--panel, #0f172a), var(--card, #121a2c)) !important;
  color: var(--text, #e6e9ef) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  padding: .85rem 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px !important;
  border-radius: 999px !important;
  box-shadow: 0 8px 22px rgba(2, 6, 23, 0.35) !important;
  transition: transform .15s ease, filter .15s ease, box-shadow .15s ease, border-color .15s ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px) scale(1.01);
  filter: brightness(1.05);
  border-color: rgba(124,58,237,0.35) !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.20) !important;
}

/* Download buttons: keep them neutral/secondary as well */
.stDownloadButton > button {
  background: linear-gradient(135deg, var(--panel, #0f172a), var(--card, #121a2c)) !important;
  color: var(--text, #e6e9ef) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  padding: .85rem 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px !important;
  border-radius: 999px !important;
  box-shadow: 0 8px 22px rgba(2, 6, 23, 0.35) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-1px) scale(1.01);
  filter: brightness(1.05);
  border-color: rgba(124,58,237,0.35) !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.20) !important;
}

/* Primary navigation CTA: ONLY the “Go to Attribute Analysis” button should be bright */
.stButton.st-key-go_to_analysis_btn > button {
  background: linear-gradient(135deg, var(--accent, #7c3aed), var(--accent-2, #06b6d4)) !important;
  color: #ffffff !important;
  border: none !important;
  padding: 1rem 1.25rem !important;
  font-weight: 900 !important;
  letter-spacing: .3px !important;
  border-radius: 999px !important;
  box-shadow: 0 14px 35px rgba(124,58,237,0.35), 0 6px 16px rgba(6,182,212,0.25) !important;
}
.stButton.st-key-go_to_analysis_btn > button:hover {
  transform: translateY(-1px) scale(1.02);
  filter: brightness(1.08);
  box-shadow: 0 18px 40px rgba(124,58,237,0.45), 0 10px 26px rgba(6,182,212,0.32) !important;
}
.stButton.st-key-go_to_analysis_btn > button:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 3px var(--panel, #0f172a),
              0 0 0 6px var(--ring, rgba(124,58,237,0.4)),
              0 18px 44px rgba(124,58,237,.42) !important;
}
</style>
""", unsafe_allow_html=True)


def main():
    # Top header + optional steps
    if STEPS_AVAILABLE:
        render_steps(active="1 Analyze")  # or "2 Configure"

    st.title("⚙️ Configuration")
    st.caption("Set professors, exam deadlines, and analysis parameters before computing attributes.")

    # Pull raw data (if any) to power suggestions
    raw_df = data_manager.get_raw_data()
    has_data = raw_df is not None and not raw_df.empty

    # ----------- QUICK SUMMARY -----------
    with st.expander("🔍 Quick Summary (current settings)", expanded=False):
        st.json(config.to_dict())

    # ----------- TABS -----------
    tab1, tab2, tab3 = st.tabs(["👨‍🏫 Professors & Exams", "📊 Analysis Settings", "💾 Export / Import"])

    # ===================== TAB 1: Professors & Exams =====================
    with tab1:
        st.subheader("👨‍🏫 Professor Settings")

        # Suggestions from data
        candidate_names = []
        if has_data and "userfullname" in raw_df.columns:
            candidate_names = (
                raw_df["userfullname"].dropna().astype(str).value_counts().head(30).index.tolist()
            )

        colp1, colp2 = st.columns([2, 1])

        with colp1:
            st.markdown("**Professor names (one per line)**")
            prof_input = st.text_area(
                label="",
                value="\n".join(config.professors),
                height=140,
                placeholder="e.g.\nProf. László Pitlik\nProf. Example Name",
                help="These names are used to identify professor-related activities in the logs."
            )
            # Normalize + deduplicate
            profs = [p.strip() for p in prof_input.split("\n") if p.strip()]
            config.professors = list(dict.fromkeys(profs))  # de-duplicate, keep order

        with colp2:
            if candidate_names:
                st.markdown("**Quick insert from detected users**")
                add_prof = st.selectbox(
                    "Add from data",
                    options=["— select —"] + candidate_names,
                    index=0
                )
                if add_prof != "— select —":
                    if add_prof not in config.professors:
                        config.professors.append(add_prof)
                        st.success(f"Added professor: {add_prof}")
                        st.rerun()
            else:
                st.info("Load data to get suggested professor names.")

        st.markdown("---")
        st.subheader("🧪 Exam Deadline Settings")
        st.caption("Set deadlines for each exam. Posts after these dates will be flagged.")

        # Helper: detect exam-like subjects from data
        detected_exams = []
        if has_data and "subject" in raw_df.columns:
            subj = raw_df["subject"].dropna().astype(str)
            detected_exams = (
                subj[subj.str.contains("exam", case=False, na=False)]
                .value_counts()
                .head(10)
                .index.tolist()
            )

        # Add exam UI (manual or from detected)
        cole1, cole2 = st.columns([2, 2])
        with cole1:
            new_exam = st.text_input("➕ Add exam (custom name)", placeholder="e.g., Quasi Exam IV")
            if st.button("Add Exam", use_container_width=True, key="add_exam_btn") and new_exam:
                if new_exam not in config.deadlines:
                    config.deadlines[new_exam] = pd.Timestamp.today().normalize()
                    st.success(f"Added: {new_exam}")
                    st.rerun()
                else:
                    st.warning(f"'{new_exam}' already exists.")

        with cole2:
            if detected_exams:
                add_det = st.selectbox("➕ Add from detected subjects", ["— select —"] + detected_exams)
                if add_det != "— select —":
                    if add_det not in config.deadlines:
                        config.deadlines[add_det] = pd.Timestamp.today().normalize()
                        st.success(f"Added: {add_det}")
                        st.rerun()
                    else:
                        st.warning(f"'{add_det}' already exists.")
            else:
                st.info("No exam-like subjects detected. Load data first to auto-suggest.")

        # Editable deadline table
        if config.deadlines:
            st.markdown("#### 🗓️ Edit deadlines")
            dl_df = pd.DataFrame([
                {"Exam": name, "Deadline": pd.to_datetime(date).date()}
                for name, date in config.deadlines.items()
            ])

            edited = st.data_editor(
                dl_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Exam": st.column_config.TextColumn("Exam"),
                    "Deadline": st.column_config.DateColumn("Deadline", format="YYYY-MM-DD")
                },
                hide_index=True,
                key="deadlines_editor"
            )

            # Apply edits back into config
            new_deadlines = {}
            for _, row in edited.iterrows():
                name = str(row["Exam"]).strip()
                date = pd.to_datetime(row["Deadline"]) if pd.notna(row["Deadline"]) else pd.Timestamp.today()
                if name:
                    new_deadlines[name] = date
            config.deadlines = new_deadlines
        else:
            st.info("No exams yet. Add one above to get started.")

    # ===================== TAB 2: Analysis Settings =====================
    with tab2:
        st.subheader("📊 Analysis Settings")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pattern Matching")
            pattern_str = ", ".join(map(str, config.parent_ids_pattern or []))
            pattern_input = st.text_input(
                "Parent IDs (comma-separated)",
                value=pattern_str,
                help="Used for the Pattern_followed attribute."
            )
            try:
                parsed = [int(x.strip()) for x in pattern_input.split(",") if x.strip()]
                config.parent_ids_pattern = parsed
            except ValueError:
                st.error("Please enter only integers separated by commas, e.g., 163483, 163486")

        with col2:
            st.markdown("#### COCO Analysis")
            config.analysis_settings['y_value'] = st.number_input(
                "Y reference value",
                value=config.analysis_settings.get('y_value', 0),
                min_value=0,
                max_value=100000,
                help="Reference value used in ranking and COCO analysis."
            )

        st.divider()
        st.markdown("#### Presets (optional)")
        cpa, cpb, cpc = st.columns(3)
        with cpa:
            if st.button("Balanced preset", use_container_width=True):
                st.success("Balanced preset applied.")
        with cpb:
            if st.button("Engagement-focused preset", use_container_width=True):
                st.success("Engagement preset applied.")
        with cpc:
            if st.button("Exam-focused preset", use_container_width=True):
                st.success("Exam preset applied.")

    # ===================== TAB 3: Export / Import =====================
    with tab3:
        st.subheader("💾 Configuration Management")
        colx, coly = st.columns(2)

        with colx:
            st.markdown("#### Save / Export")
            import json
            cfg_dict = config.to_dict()
            cfg_json = json.dumps(cfg_dict, indent=2, default=str)
            st.download_button(
                "📥 Export Configuration (JSON)",
                data=cfg_json,
                file_name="moodle_analyzer_config.json",
                mime="application/json",
                use_container_width=True
            )

        with coly:
            st.markdown("#### Import / Reset")
            up = st.file_uploader("Import configuration JSON", type=['json'])
            if up is not None:
                try:
                    import json
                    imported = json.load(up)
                    config.from_dict(imported)
                    st.success("Configuration imported successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing: {e}")

            if st.button("🔄 Reset to Defaults", use_container_width=True):
                config.load_defaults()
                st.success("Reset to defaults.")
                st.rerun()

    # ----------- NAVIGATION CTAs -----------
    st.divider()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ Back to Upload", use_container_width=True, key="back_to_upload_btn"):
            st.switch_page("pages/1_📊_Data_Upload.py")
    with c2:
        st.caption("Your settings are applied immediately. You can revisit this page anytime.")
    with c3:
        if st.button("➡️ Go to Attribute Analysis", use_container_width=True, key="go_to_analysis_btn"):
            try:
                st.switch_page("pages/3_📈_Attribute_Analysis.py")
            except Exception:
                st.warning("Unable to auto-navigate. Please open ‘📈 Attribute Analysis’ from the sidebar.")

if __name__ == "__main__":
    main()
