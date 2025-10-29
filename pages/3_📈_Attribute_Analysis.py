# pages/3_📈_Attribute_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pathlib

from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.attribute_calculations import (
    ATTRIBUTE_FUNCS, activity_attrs, engagement_attrs,
    content_attrs, exam_attrs, available_attributes,
    to_dt
)

# ---------- Safe initialization ----------
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

# ---------- Baseline design system (same as Home / Upload) ----------
st.markdown("""
<style>
:root{
  --bg: #000B18;
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
  font-size:2   rem;
  display:flex; align-items:center; gap:.6rem;
  font-weight: 800; letter-spacing:-0.02em;
  margin: 1.2rem 0 .8rem 0; color: var(--text);
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

/* Default buttons on this page = secondary/dark */
.stButton > button:not([data-testid="baseButton-primary"]) {
  background: linear-gradient(135deg, var(--panel), var(--card)) !important;
  color: var(--text) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  padding: .9rem 1.1rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px !important;
  border-radius: 999px !important;
  box-shadow: 0 8px 22px rgba(2,6,23,0.35) !important;
  transition: transform .15s ease, filter .15s ease, box-shadow .15s ease, border-color .15s ease !important;
}
.stButton > button:not([data-testid="baseButton-primary"]):hover {
  transform: translateY(-1px) scale(1.01);
  filter: brightness(1.05);
  border-color: rgba(124,58,237,0.35) !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.20) !important;
}

/* Download buttons secondary too */
.stDownloadButton > button {
  background: linear-gradient(135deg, var(--panel), var(--card)) !important;
  color: var(--text) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  padding: .9rem 1.1rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px !important;
  border-radius: 999px !important;
  box-shadow: 0 8px 22px rgba(2,6,23,0.35) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-1px) scale(1.01);
  filter: brightness(1.05);
  border-color: rgba(124,58,237,0.35) !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.20) !important;
}

/* Primary CTAs: ONLY specific primary buttons */
.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: #fff !important;
  border: none !important;
  padding: .9rem 1.1rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px !important;
  border-radius: 999px !important;
  box-shadow: 0 10px 28px rgba(124,58,237,0.35) !important;
  transition: transform .15s ease, filter .15s ease, box-shadow .15s ease !important;
}

.stButton > button[data-testid="baseButton-primary"]:hover {
  transform: translateY(-1px) scale(1.01);
  filter: brightness(1.05);
  box-shadow: 0 16px 42px rgba(124,58,237,0.42) !important;
}

.stButton > button[data-testid="baseButton-primary"]:focus-visible {
  outline: none !important;
  box-shadow: 
    0 0 0 3px var(--panel),
    0 0 0 6px var(--ring),
    0 16px 42px rgba(124,58,237,0.42) !important;
}
/* Expander polish */
.streamlit-expanderHeader { font-weight: 700; color: var(--text); }
.stExpander { border: 1px solid rgba(148,163,184,0.12); border-radius: var(--radius); }

/* Divider */
hr{ border: none; border-top:1px solid rgba(148,163,184,0.15); margin: 1.2rem 0; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
""", unsafe_allow_html=True)


def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config

    # ---------- HEADER ----------
    st.markdown('<div class="section-title"></span><span>Attribute Analysis</span></div>', unsafe_allow_html=True)
    st.caption("Compute and analyze student attributes. Select attributes and generate the OAM.")

    # Check if data is available
    raw_data = data_manager.get_raw_data()
    if raw_data is None:
        st.warning("📊 Please upload data first on the Data Upload page.")
        return

    # Use configuration from ConfigManager
    PROFESSORS = config.professors
    DEADLINES = config.deadlines
    PARENT_IDS_PATTERN = config.parent_ids_pattern

    # Prepare data (exclude professors)
    df_all = raw_data.copy()
    df_all["userfullname"] = df_all["userfullname"].astype(str)
    df = df_all[~df_all["userfullname"].isin(PROFESSORS)].copy()

    
    col_meta1, col_meta2 = st.columns(2)
    
    st.markdown("""
    <style>
    .metric-box {
        background: rgba(124, 58, 237, 0.15); /* soft purple highlight */
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #e6e9ef;
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.2);
    }
    .metric-box span {
        font-size: 28px;
        color: #a78bfa; /* lighter purple for emphasis */
    }
    </style>
    """, unsafe_allow_html=True)
    
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
    
    
        # Attribute selection UI
    render_attribute_selection_ui()

    # Compute attributes CTA (bright)
    # Update the compute attributes button
    if st.button("🚀 Compute Attributes", type="primary", use_container_width=True, key="compute_attributes_btn"):
        compute_and_display_attributes(df, df_all, data_manager, config)

    # After compute
    student_attributes = data_manager.get_student_attributes()
    if student_attributes is not None:
        display_hybrid_layout(student_attributes, data_manager)

    # Visualization section
    if data_manager.get_student_attributes() is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        display_graph_section(data_manager.get_student_attributes())

    # Navigation CTA (bright)
    # In the navigation section of your code, update the button to:
    if data_manager.get_student_attributes() is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button(
                "🏆 Proceed to Ranking",
                type="primary",  # Add this line
                use_container_width=True,
                help="Navigate to the ranking page with computed attributes",
                key="proceed_to_ranking_btn"
            ):
                st.switch_page("pages/4_🏆_Ranking.py")
    

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

    # Selected count card
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding:.7rem; margin:.3rem 0 1rem 0;">
          <div style="color:var(--muted); font-weight:700;">Selected attributes</div>
          <div style="font-size:1.35rem; font-weight:900;">{len(st.session_state.selected_attributes)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
    st.markdown("<hr>", unsafe_allow_html=True)
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
                        "deadline_exceeded_posts_Quasi_exam_I": "Quasi Exam I",
                        "deadline_exceeded_posts_Quasi_exam_II": "Quasi Exam II",
                        "deadline_exceeded_posts_Quasi_exam_III": "Quasi Exam III",
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

    st.success("✅ All attributes have been computed successfully!")

    # Fill NaN values and sort
    oam_combined = oam_combined.fillna(0)
    fixed_cols = ["userid", "userfullname"]
    attr_cols = [c for c in oam_combined.columns if c not in fixed_cols]
    attr_cols_sorted = sorted(attr_cols)
    oam_combined = oam_combined[fixed_cols + attr_cols_sorted]
    oam_combined = oam_combined.sort_values("userfullname")

    # Store in session state
    data_manager.store_student_attributes(oam_combined)


def display_hybrid_layout(oam_combined, data_manager):
    """Display attributes in hybrid layout (categories + combined)"""

    # Category tables
    activity_table = create_category_table(oam_combined, activity_attrs, "Activity")
    engagement_table = create_category_table(oam_combined, engagement_attrs, "Engagement")
    content_table = create_category_table(oam_combined, content_attrs, "Content")
    exam_table = create_category_table(oam_combined, exam_attrs, "Exam")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🚀 Activity",
        "💬 Engagement",
        "📝 Content",
        "📋 Exams"
    ])

    with tab1:
        display_overview_dashboard(oam_combined, activity_table, engagement_table, content_table, exam_table)

    with tab2:
        display_category_table(activity_table, "Activity Metrics", "Measures posting frequency and consistency")

    with tab3:
        display_category_table(engagement_table, "Engagement Metrics", "Measures interaction quality and response patterns")

    with tab4:
        display_category_table(content_table, "Content Analysis", "Analyzes content quality and relevance")

    with tab5:
        display_category_table(exam_table, "Exam Performance", "Tracks exam-related behavior and deadlines")

    # Combined OAM (expandable)
    with st.expander("🔗 Combined Object Attribute Matrix (For COCO Analysis)", expanded=False):
        st.markdown("**Full OAM with all attributes - Use this for COCO analysis**")
        st.dataframe(oam_combined, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_data = oam_combined.to_csv(index=False)
            st.download_button(
                "📥 Download Full OAM (CSV)",
                csv_data,
                "full_oam_matrix.csv",
                "text/csv",
                use_container_width=True,
                key="download_full_oam_btn"
            )


def create_category_table(oam_combined, category_attrs, category_name):
    available_attrs = [attr for attr in category_attrs if attr in oam_combined.columns]
    if available_attrs:
        return oam_combined[["userid", "userfullname"] + available_attrs]
    return oam_combined[["userid", "userfullname"]].copy()


def display_overview_dashboard(oam_combined, activity_table, engagement_table, content_table, exam_table):
    st.subheader("📈 Analysis Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(oam_combined))
    with col2:
        st.metric("Total Attributes", len(oam_combined.columns) - 2)

    st.subheader("📋 Attribute Summary by Category")
    stats_data = {
        "Category": ["Activity", "Engagement", "Content", "Exam"],
        "Attributes": [
            len(activity_table.columns) - 2,
            len(engagement_table.columns) - 2,
            len(content_table.columns) - 2,
            len(exam_table.columns) - 2
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True, key="stats_df")

    with st.expander("🔍 Quick Data Preview", expanded=False):
        st.dataframe(oam_combined.head(10), use_container_width=True, key="data_preview_df")


def display_category_table(category_table, title, description):
    st.subheader(title)
    st.caption(description)

    if len(category_table.columns) > 2:
        st.dataframe(category_table, use_container_width=True, key=f"{title.lower()}_df")

        csv_data = category_table.to_csv(index=False)
        category_name = title.lower().replace(" ", "_")
        st.download_button(
            f"📥 Download {title} (CSV)",
            csv_data,
            f"{category_name}_attributes.csv",
            "text/csv",
            use_container_width=True,
            key=f"download_{category_name}_btn"
        )
    else:
        st.info(f"No {title.lower()} attributes selected or computed.")


def display_graph_section(oam_combined):
    st.header("📈 Attribute & Student Visualizations")

    fixed_cols = ["userid", "userfullname"]
    attribute_cols = [col for col in oam_combined.columns if col not in fixed_cols]

    if not attribute_cols:
        st.warning("No attributes available for visualization. Please compute attributes first.")
        return

    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "📊 Attribute Distribution Analysis",
            "👥 Student Performance Comparison",
            "🔥 Top Performers by Attribute",
            "📈 Student Attribute Profile",
            "🌐 Correlation Heatmap",
            "📋 Category-wise Analysis"
        ],
        key="viz_type_select"
    )

    viz_container = st.container()

    with viz_container:
        if viz_type == "📊 Attribute Distribution Analysis":
            display_attribute_distribution(oam_combined, attribute_cols)

        elif viz_type == "👥 Student Performance Comparison":
            display_student_comparison(oam_combined, attribute_cols)

        elif viz_type == "🔥 Top Performers by Attribute":
            display_top_performers(oam_combined, attribute_cols)

        elif viz_type == "📈 Student Attribute Profile":
            display_student_profile(oam_combined, attribute_cols)

        elif viz_type == "🌐 Correlation Heatmap":
            display_correlation_heatmap(oam_combined, attribute_cols)

        elif viz_type == "📋 Category-wise Analysis":
            display_category_analysis(oam_combined)


def display_attribute_distribution(oam_combined, attribute_cols):
    st.subheader("📊 Attribute Distribution Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_attribute = st.selectbox(
            "Select Attribute to Analyze",
            attribute_cols,
            key="attr_dist_select"
        )

        if selected_attribute:
            attr_data = oam_combined[selected_attribute]
            stats = {
                "Mean": attr_data.mean(),
                "Median": attr_data.median(),
                "Std Dev": attr_data.std(),
                "Min": attr_data.min(),
                "Max": attr_data.max()
            }

            st.metric("Average", f"{stats['Mean']:.2f}")
            st.metric("Median", f"{stats['Median']:.2f}")
            st.metric("Std Deviation", f"{stats['Std Dev']:.2f}")

    with col2:
        if selected_attribute:
            fig = px.histogram(
                oam_combined,
                x=selected_attribute,
                title=f"Distribution of {selected_attribute.replace('_', ' ').title()}",
                nbins=20
            )
            fig.update_layout(
                xaxis_title=selected_attribute.replace('_', ' ').title(),
                yaxis_title="Number of Students",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="dist_histogram")

            fig_box = px.box(
                oam_combined,
                y=selected_attribute,
                title=f"Box Plot - {selected_attribute.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_box, use_container_width=True, key="dist_boxplot")


def display_student_comparison(oam_combined, attribute_cols):
    st.subheader("👥 Student Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        selected_students = st.multiselect(
            "Select Students to Compare",
            options=oam_combined["userfullname"].tolist(),
            default=oam_combined["userfullname"].head(5).tolist(),
            key="student_comparison_multiselect"
        )

    with col2:
        selected_attributes = st.multiselect(
            "Select Attributes for Comparison",
            options=attribute_cols,
            default=attribute_cols[:5] if len(attribute_cols) >= 3 else attribute_cols,
            key="attr_comparison_multiselect"
        )

    if selected_students and selected_attributes:
        comparison_data = oam_combined[oam_combined["userfullname"].isin(selected_students)]

        if len(selected_attributes) >= 3:
            fig_radar = create_radar_chart(comparison_data, selected_students, selected_attributes)
            st.plotly_chart(fig_radar, use_container_width=True, key="comparison_radar")

        fig_bar = create_attribute_comparison_bar(comparison_data, selected_students, selected_attributes)
        st.plotly_chart(fig_bar, use_container_width=True, key="comparison_bar")


def create_radar_chart(comparison_data, students, attributes):
    fig = go.Figure()

    normalized_data = comparison_data.copy()
    for attr in attributes:
        max_val = normalized_data[attr].max()
        if max_val > 0:
            normalized_data[attr] = normalized_data[attr] / max_val

    for student in students:
        student_data = normalized_data[normalized_data["userfullname"] == student]
        values = student_data[attributes].iloc[0].tolist()
        values.append(values[0])  # Close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=attributes + [attributes[0]],
            fill='toself',
            name=student
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Student Comparison Radar Chart"
    )

    return fig


def create_attribute_comparison_bar(comparison_data, students, attributes):
    melt_data = comparison_data.melt(
        id_vars=["userfullname"],
        value_vars=attributes,
        var_name="Attribute",
        value_name="Value"
    )

    fig = px.bar(
        melt_data,
        x="userfullname",
        y="Value",
        color="Attribute",
        barmode="group",
        title="Student Attribute Comparison"
    )

    fig.update_layout(
        xaxis_title="Students",
        yaxis_title="Attribute Value",
        showlegend=True
    )

    return fig


def display_top_performers(oam_combined, attribute_cols):
    st.subheader("🔥 Top Performers by Attribute")

    selected_attribute = st.selectbox(
        "Select Attribute for Ranking",
        attribute_cols,
        key="top_perf_select"
    )

    top_n = st.slider("Number of Top Students to Show", 5, 20, 10, key="top_n_slider")

    if selected_attribute:
        top_students = oam_combined.nlargest(top_n, selected_attribute)[["userfullname", selected_attribute]]

        fig = px.bar(
            top_students,
            y="userfullname",
            x=selected_attribute,
            orientation='h',
            title=f"Top {top_n} Students - {selected_attribute.replace('_', ' ').title()}",
            color=selected_attribute,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            yaxis_title="Student",
            xaxis_title=selected_attribute.replace('_', ' ').title(),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, key="top_performers_chart")
        st.dataframe(top_students, use_container_width=True, key="top_performers_df")


def display_student_profile(oam_combined, attribute_cols):
    st.subheader("👤 Student Profile")

    selected_student = st.selectbox(
        "Select Student",
        oam_combined["userfullname"].tolist(),
        key="student_profile_select"
    )

    if selected_student:
        student_data = oam_combined[oam_combined["userfullname"] == selected_student].iloc[0]

        st.markdown(f"### 📊 Profile for: **{selected_student}**")

        profile_data = []
        for attr in attribute_cols:
            profile_data.append({
                'Attribute': attr.replace('_', ' ').title(),
                'Score': f"{student_data[attr]:.2f}",
                'Class Average': f"{oam_combined[attr].mean():.2f}",
                'Status': '✅ Above Avg' if student_data[attr] > oam_combined[attr].mean() else '📊 Below Avg'
            })

        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, height=500, key="student_profile_df")

        above_avg = sum(1 for attr in attribute_cols
                        if student_data[attr] > oam_combined[attr].mean())

        st.info(f"**Summary:** {above_avg} out of {len(attribute_cols)} attributes are above class average")


def display_correlation_heatmap(oam_combined, attribute_cols):
    st.subheader("🌐 Attribute Correlation Heatmap")

    if len(attribute_cols) < 2:
        st.warning("Need at least 2 attributes for correlation analysis")
        return

    corr_matrix = oam_combined[attribute_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Attribute Correlation Heatmap"
    )

    fig.update_layout(
        xaxis_title="Attributes",
        yaxis_title="Attributes",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

    with st.expander("💡 Correlation Interpretation Guide"):
        st.markdown("""
        **Correlation Values Meaning:**
        - **+1.0**: Perfect positive correlation
        - **+0.7 to +1.0**: Strong positive correlation  
        - **+0.3 to +0.7**: Moderate positive correlation
        - **-0.3 to +0.3**: Weak or no correlation
        - **-0.7 to -0.3**: Moderate negative correlation
        - **-1.0**: Perfect negative correlation
        """)


def display_category_analysis(oam_combined):
    st.subheader("📋 Category-wise Attribute Analysis")

    activity_cols = [col for col in oam_combined.columns if col in activity_attrs]
    engagement_cols = [col for col in oam_combined.columns if col in engagement_attrs]
    content_cols = [col for col in oam_combined.columns if col in content_attrs]
    exam_cols = [col for col in oam_combined.columns if col in exam_attrs]

    categories = {
        "Activity": activity_cols,
        "Engagement": engagement_cols,
        "Content": content_cols,
        "Exam": exam_cols
    }
    categories = {k: v for k, v in categories.items() if v}

    if not categories:
        st.warning("No categorized attributes available")
        return

    selected_category = st.selectbox(
        "Select Category",
        list(categories.keys()),
        key="category_select"
    )

    if selected_category and categories[selected_category]:
        category_cols = categories[selected_category]
        category_avg = oam_combined[category_cols].mean()

        fig = px.bar(
            x=category_cols,
            y=category_avg.values,
            title=f"{selected_category} Category - Average Scores",
            labels={'x': 'Attributes', 'y': 'Average Score'},
            color=category_avg.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="category_avg_chart")

        st.subheader(f"🏆 Top Performers - {selected_category} Category")

        if category_cols:
            oam_combined[f'{selected_category.lower()}_total'] = oam_combined[category_cols].sum(axis=1)
            top_students = oam_combined.nlargest(10, f'{selected_category.lower()}_total')[['userfullname', f'{selected_category.lower()}_total']]

            fig_top = px.bar(
                top_students,
                y='userfullname',
                x=f'{selected_category.lower()}_total',
                orientation='h',
                title=f"Top 10 Students - {selected_category} Category",
                color=f'{selected_category.lower()}_total',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_top, use_container_width=True, key="category_top_chart")


if __name__ == "__main__":
    main()
