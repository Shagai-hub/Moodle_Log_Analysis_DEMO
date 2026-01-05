# pages/3_📈_Attribute_Analysis.py
import streamlit as st
import pandas as pd

from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.attribute_calculations import (
    ATTRIBUTE_FUNCS, activity_attrs, engagement_attrs,
    content_attrs, exam_attrs, available_attributes,
    to_dt
)
from assets.ui_components import (
    apply_theme,
    centered_page_button,
    divider,
    info_panel,
    page_header,
    section_header,
    nav_footer,
)
from utils.ui_steps import render_steps
# ---------- Safe initialization ----------
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
        "Attribute Analysis",
        "Compute and analyse student attributes. Select metrics and generate the OAM.",
        icon="📈",
        align="left",
        compact=True,
    )

    # Check if data is available
    raw_data = data_manager.get_raw_data()
    if raw_data is None:
        st.warning("📊 Please upload data first on the Data Upload page.")
        divider()
        nav_footer(
            back={
                "label": "⬅️ Back to Configuration",
                "page": "pages/2_⚙️_Configuration.py",
                "key": "nav_back_to_configuration_missing_data",
                "fallback": "⚙️ Configuration",
            },
            message="Upload your discussion logs to unlock attribute analysis.",
            forward=None,
        )
        return

    # Use configuration from ConfigManager
    PROFESSORS = config.professors
    DEADLINES = config.deadlines
    PARENT_IDS_PATTERN = config.parent_ids_pattern

    # Prepare data (exclude professors)
    df_all = raw_data.copy()
    df_all["userfullname"] = df_all["userfullname"].astype(str)
    df = df_all[~df_all["userfullname"].isin(PROFESSORS)].copy()

    section_header("Dataset snapshot", icon="🗂️", tight=True)
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"""
        <div class="metric-box">
            📊 Posts considered<br><span>{len(df)}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_meta2:
        st.markdown(f"""
        <div class="metric-box">
            👨‍🏫 Professors<br><span>{', '.join(PROFESSORS) if PROFESSORS else '—'}</span>
        </div>
        """, unsafe_allow_html=True)

    section_header("Select attributes", icon="🎛️")

    # Attribute selection UI
    render_attribute_selection_ui()

    # Compute attributes CTA (bright)
    # Update the compute attributes button
    if st.button("🚀 Compute Attributes", type="primary", use_container_width=True, key="compute_attributes_btn"):
        compute_and_display_attributes(df, df_all, data_manager, config)

    # After compute
    student_attributes = data_manager.get_student_attributes()
    if student_attributes is not None:
        display_hybrid_layout(student_attributes)
        centered_page_button(
            "Visualizations",
            "pages/6_📊_Visualizations.py",
            key="pulse",
            icon="📊",
            help="Open interactive dashboards built from the computed attributes.",
            fallback="📊 Visualizations",
            button_type="secondary",
        )


    forward_spec = None
    if data_manager.get_student_attributes() is not None:
        forward_spec = {
            "label": "🏆 Proceed to Ranking",
            "page": "pages/4_🏆_Ranking.py",
            "key": "pulse1",
            "fallback": "🏆 Ranking",
            "help": "Navigate to the ranking page with computed attributes",
        }

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to Configuration",
            "page": "pages/2_⚙️_Configuration.py",
            "key": "nav_back_to_configuration",
            "fallback": "⚙️ Configuration",
        },
        message="Review configuration or upload settings anytime—your selections stay in session.",
        forward=forward_spec,
    )

def render_attribute_selection_ui():
    """Render the attribute selection interface"""
    if "selected_attributes" not in st.session_state:
        st.session_state.selected_attributes = []

    # Create attribute key mapping with unique keys
    attr_key_map = {}
    for i, attr in enumerate(activity_attrs):
        attr_key_map[attr] = f"activity_{attr}_{i}"
    for i, attr in enumerate(engagement_attrs):
        attr_key_map[attr] = f"engagement_{attr}_{i}"
    for i, attr in enumerate(content_attrs):
        attr_key_map[attr] = f"content_{attr}_{i}"
    for i, attr in enumerate(exam_attrs):
        attr_key_map[attr] = f"exam_{attr}_{i}"

    # Helper functions for select all/clear all
    def select_all():
        st.session_state.selected_attributes = available_attributes.copy()
        for attr in available_attributes:
            key = attr_key_map.get(attr)
            if key is not None:
                st.session_state[key] = True

    def clear_all():
        st.session_state.selected_attributes = []
        for key in attr_key_map.values():
            st.session_state[key] = False

    # Attribute descriptions
    with st.expander("ℹ️ Attribute Descriptions", expanded=False):
        st.markdown("""
        **Activity Metrics:** Posting frequency, consistency, and engagement patterns  
        **Engagement Metrics:** Interaction quality and response patterns  
        **Content Analysis:** Content quality, length, and relevance  
        **Exam Performance:** Exam-related posting behavior and deadline compliance  
        """)
        
    count = len(st.session_state.selected_attributes)
    # Selected count card
    st.markdown(
    f"""
    <div style='text-align:center; font-size:2.5rem; font-weight:700; color:#ADD8E6; margin-top:0.5rem;'>
        Selected attributes: {count}
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("")

    # Category expanders for attribute selection
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📊 Activity Metrics", expanded=False):
            for attr in activity_attrs:
                if attr in available_attributes:
                    key = attr_key_map[attr]
                    initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
                    checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
                    update_selected_attributes(attr, checked)

        with st.expander("💬 Engagement Metrics", expanded=False):
            for attr in engagement_attrs:
                if attr in available_attributes:
                    key = attr_key_map[attr]
                    initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
                    checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
                    update_selected_attributes(attr, checked)

    with col2:
        with st.expander("📝 Content Analysis", expanded=False):
            st.info("⚠️ Some ML-based attributes may take longer to compute", icon="⚙️")
            for attr in content_attrs:
                if attr in available_attributes:
                    key = attr_key_map[attr]
                    initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
                    checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
                    update_selected_attributes(attr, checked)

        with st.expander("📋 Exam Performance", expanded=False):
            for attr in exam_attrs:
                if attr in available_attributes:
                    key = attr_key_map[attr]
                    initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
                    checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
                    update_selected_attributes(attr, checked)

    # Selection controls
    divider()
    colsa, colsb = st.columns([1, 1])
    with colsa:
        st.button("✅ Select All", on_click=select_all, use_container_width=True, key="select_all_btn")
    with colsb:
        st.button("❌ Clear All", on_click=clear_all, use_container_width=True, key="clear_all_btn")


def update_selected_attributes(attr, checked):
    """Update the selected attributes list based on checkbox state"""
    if checked and attr not in st.session_state.selected_attributes:
        st.session_state.selected_attributes.append(attr)
    if not checked and attr in st.session_state.selected_attributes:
        st.session_state.selected_attributes.remove(attr)


def compute_and_display_attributes(df, df_all, data_manager, config):
    """Compute attributes and display in hybrid layout"""
    if not st.session_state.selected_attributes:
        st.error("❌ Please select at least one attribute to compute.")
        return

    PROFESSORS = config.professors
    DEADLINES = config.deadlines
    PARENT_IDS_PATTERN = config.parent_ids_pattern

    students = df[["userid", "userfullname"]].drop_duplicates().sort_values("userfullname").reset_index(drop=True)
    oam_combined = students.copy()

    with st.spinner("⏳ Computing selected attributes..."):
        for i, attr in enumerate(st.session_state.selected_attributes):
            try:
                func = ATTRIBUTE_FUNCS[attr]

                # Handle functions that need additional parameters
                if attr == "total_replies_to_professor":
                    prof_name = PROFESSORS[0] if PROFESSORS else "professor_1"
                    result = func(df, df_all, prof_name)

                elif attr == "topic_relevance_score":
                    prof_name = PROFESSORS[0] if PROFESSORS else "professor_1"
                    result = func(df, df_all, prof_name)

                elif attr in ["engagement_rate", "avg_reply_time"]:
                    result = func(df, df_all)

                elif attr.startswith("deadline_exceeded_posts_"):
                    exam_mapping = {
                        "deadline_exceeded_posts_Quasi_exam_I": "Quasi_exam_I",
                        "deadline_exceeded_posts_Quasi_exam_II": "Quasi_exam_II",
                        "deadline_exceeded_posts_Quasi_exam_III": "Quasi_exam_III",
                    }
                    if attr in exam_mapping:
                        exam_name = exam_mapping[attr]
                        if exam_name in DEADLINES:
                            result = func(df)
                        else:
                            st.warning(f"Deadline for {exam_name} not found in configuration")
                            continue
                    else:
                        st.warning(f"No mapping found for attribute: {attr}")
                        continue

                elif attr == "Pattern_followed_quasi_exam_i":
                    result = func(df)

                else:
                    result = func(df)

                # Merge results
                if result is not None and not result.empty:
                    key_cols = ["userid", "userfullname"]
                    result_cols = [c for c in result.columns if c not in key_cols]
                    if result_cols:
                        oam_combined = oam_combined.merge(
                            result.drop(columns=["userfullname"], errors='ignore'),
                            on="userid",
                            how="left",
                        )
                    else:
                        st.warning(f"Attribute {attr} produced no value column; skipping")
                else:
                    oam_combined[attr] = 0

            except Exception as e:
                st.error(f"Error computing {attr}: {e}")
                oam_combined[attr] = 0
    st.snow()
    st.toast("✅ All attributes have been computed successfully!")

    # Fill NaN values and sort
    oam_combined = oam_combined.fillna(0)
    fixed_cols = ["userid", "userfullname"]
    attr_cols = [c for c in oam_combined.columns if c not in fixed_cols]
    attr_cols_sorted = sorted(attr_cols)
    oam_combined = oam_combined[fixed_cols + attr_cols_sorted]
    oam_combined = oam_combined.sort_values("userfullname")

    # Store in session state
    data_manager.store_student_attributes(oam_combined)


def display_hybrid_layout(oam_combined):
    """Lightweight preview of the computed attribute matrix."""

    section_header("Attribute Matrix Preview", icon="🗂️", tight=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Students", len(oam_combined))
    with col2:
        st.metric("Attributes", max(len(oam_combined.columns) - 2, 0))

    st.caption("Previewing the first 20 rows of the computed Object Attribute Matrix.")
    st.dataframe(oam_combined.head(20), use_container_width=True, hide_index=True, key="oam_preview_df")
    st.caption(f"Full matrix size: {oam_combined.shape[0]} rows × {oam_combined.shape[1]} columns")

    csv_data = oam_combined.to_csv(index=False)
    st.download_button(
        "📥 Download Full OAM (CSV)",
        csv_data,
        "full_oam_matrix.csv",
        "text/csv",
        use_container_width=True,
        key="download_full_oam_btn",
    )

if __name__ == "__main__":
    main()


