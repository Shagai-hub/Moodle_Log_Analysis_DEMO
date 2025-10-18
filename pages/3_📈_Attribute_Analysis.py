import streamlit as st
import pandas as pd
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.attribute_calculations import (
    ATTRIBUTE_FUNCS, activity_attrs, engagement_attrs, 
    content_attrs, exam_attrs, available_attributes,
    to_dt
)

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    # Check if data is available
    raw_data = data_manager.get_raw_data()
    if raw_data is None:
        st.warning("📊 Please upload data first on the Data Upload page.")
        return
    
    st.title("📈 Attribute Analysis")
    st.markdown("Compute and analyze student attributes. Select desired attributes and generate the Object Attribute Matrix (OAM).")
    
    # Use configuration from ConfigManager
    PROFESSORS = config.professors
    DEADLINES = config.deadlines
    PARENT_IDS_PATTERN = config.parent_ids_pattern
    
    # Prepare data (exclude professors)
    df_all = raw_data.copy()
    df_all["userfullname"] = df_all["userfullname"].astype(str)
    df = df_all[~df_all["userfullname"].isin(PROFESSORS)].copy()
    
    st.write(f"📊 Analyzing dataset with **{len(df)}** student posts")
    st.write(f"👨‍🏫 Professors: {', '.join(PROFESSORS)}")
    
    # Attribute selection UI (similar to your current code)
    render_attribute_selection_ui()
    
    # Compute attributes when requested
    if st.button("🚀 Compute Selected Attributes", use_container_width=True):
        compute_and_display_attributes(df, df_all, data_manager, config)

def render_attribute_selection_ui():
    """Render the attribute selection interface"""
    # Initialize session state for selected attributes
    if "selected_attributes" not in st.session_state:
        st.session_state.selected_attributes = []
    
    # Create attribute key mapping (same as your current code)
    attr_key_map = {}
    for attr in activity_attrs:
        attr_key_map[attr] = f"activity_{attr}"
    for attr in engagement_attrs:
        attr_key_map[attr] = f"engagement_{attr}"
    for attr in content_attrs:
        attr_key_map[attr] = f"content_{attr}"
    for attr in exam_attrs:
        attr_key_map[attr] = f"exam_{attr}"
    
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
    with st.expander("ℹ️ Attribute Descriptions", expanded=True):
        st.markdown("""
        **Activity Metrics:** Posting frequency, consistency, and engagement patterns  
        **Engagement Metrics:** Interaction quality and response patterns  
        **Content Analysis:** Content quality, length, and relevance  
        **Exam Performance:** Exam-related posting behavior and deadline compliance  
        """)
    
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
            st.warning("⚠️ ML-based attributes may take longer to compute")
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
    st.markdown("---")
    
    # --- Row 1: Action Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("✅ Select All", on_click=select_all, use_container_width=True)
    with col2:
        st.button("❌ Clear All", on_click=clear_all, use_container_width=True)
    
    # --- Row 2: Selection Indicator ---
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 6px;
            margin-top: 10px;
            margin-bottom: 10px;
            background-color: #262730;
            color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        ">
            <strong>📊 Selected</strong><br>
            <span style="font-size: 18px; font-weight: bold;">{len(st.session_state.selected_attributes)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )



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
    
    # Get configuration
    PROFESSORS = config.professors
    DEADLINES = config.deadlines
    PARENT_IDS_PATTERN = config.parent_ids_pattern
    
    students = df[["userid", "userfullname"]].drop_duplicates().sort_values("userfullname").reset_index(drop=True)
    oam_combined = students.copy()
    
    st.spinner("⏳ Computing selected attributes, please wait...")
    
    # Compute all selected attributes
    progress_bar = st.progress(0)
    for i, attr in enumerate(st.session_state.selected_attributes):
        progress_bar.progress(int((i / len(st.session_state.selected_attributes)) * 100))
        
        try:
            func = ATTRIBUTE_FUNCS[attr]
            
            # Handle functions that need additional parameters
            if attr == "total_replies_to_professor":
                # Use first professor from config
                prof_name = PROFESSORS[0] if PROFESSORS else "professor_1"
                result = func(df, df_all, prof_name)
            elif attr == "topic_relevance_score":
                # Use first professor from config
                prof_name = PROFESSORS[0] if PROFESSORS else "professor_1"
                result = func(df, df_all, prof_name)
            elif attr in ["engagement_rate", "avg_reply_time"]:
                # These functions now use config internally, so only need 2 arguments
                result = func(df, df_all)
            elif attr.startswith("deadline_exceeded_posts_"):
                # Map attribute names to exact exam names in configuration
                exam_mapping = {
                    "deadline_exceeded_posts_Quasi_exam_I": "Quasi Exam I",
                    "deadline_exceeded_posts_Quasi_exam_II": "Quasi Exam II", 
                    "deadline_exceeded_posts_Quasi_exam_III": "Quasi Exam III"
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
                # Use parent IDs from config
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
                        on="userid", how="left"
                    )
                else:
                    st.warning(f"Attribute {attr} produced no value column; skipping")
            else:
                oam_combined[attr] = 0
                
        except Exception as e:
            st.error(f"Error computing {attr}: {e}")
            oam_combined[attr] = 0
    
    progress_bar.progress(100)
    
    # Fill NaN values and sort
    oam_combined = oam_combined.fillna(0)
    fixed_cols = ["userid", "userfullname"]
    attr_cols = [c for c in oam_combined.columns if c not in fixed_cols]
    attr_cols_sorted = sorted(attr_cols)
    oam_combined = oam_combined[fixed_cols + attr_cols_sorted]
    oam_combined = oam_combined.sort_values("userfullname")
    
    # Store in session state
    data_manager.store_student_attributes(oam_combined)
    
    st.success(f"✅ Computed {len(attr_cols)} attributes for {len(oam_combined)} students!")
    
    # Display in hybrid layout
    display_hybrid_layout(oam_combined, data_manager)

def display_hybrid_layout(oam_combined, data_manager):
    """Display attributes in hybrid layout (categories + combined)"""
    
    # Create category tables
    activity_table = create_category_table(oam_combined, activity_attrs, "Activity")
    engagement_table = create_category_table(oam_combined, engagement_attrs, "Engagement")
    content_table = create_category_table(oam_combined, content_attrs, "Content")
    exam_table = create_category_table(oam_combined, exam_attrs, "Exam")
    
    # Tabbed interface
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
    
    # Combined OAM for COCO (expandable)
    with st.expander("🔗 Combined Object Attribute Matrix (For COCO Analysis)", expanded=False):
        st.markdown("**Full OAM with all attributes - Use this for COCO analysis**")
        st.dataframe(oam_combined, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = oam_combined.to_csv(index=False)
            st.download_button(
                "📥 Download Full OAM (CSV)",
                csv_data,
                "full_oam_matrix.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.button("🎯 Proceed to Ranking", use_container_width=True, 
                     on_click=lambda: st.session_state.update({"proceed_to_ranking": True}))

def create_category_table(oam_combined, category_attrs, category_name):
    """Create a table for a specific category"""
    available_attrs = [attr for attr in category_attrs if attr in oam_combined.columns]
    if available_attrs:
        return oam_combined[["userid", "userfullname"] + available_attrs]
    return oam_combined[["userid", "userfullname"]].copy()

def display_overview_dashboard(oam_combined, activity_table, engagement_table, content_table, exam_table):
    """Display overview dashboard with summary metrics"""
    st.subheader("📈 Analysis Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(oam_combined))
    with col2:
        st.metric("Total Attributes", len(oam_combined.columns) - 2)
    with col3:
        activity_count = len(activity_table.columns) - 2
        st.metric("Activity Metrics", activity_count)
    with col4:
        engagement_count = len(engagement_table.columns) - 2
        st.metric("Engagement Metrics", engagement_count)
    
    # Quick stats
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
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Data preview
    with st.expander("🔍 Quick Data Preview", expanded=False):
        st.dataframe(oam_combined.head(10), use_container_width=True)

def display_category_table(category_table, title, description):
    """Display a category table with download option"""
    st.subheader(title)
    st.caption(description)
    
    if len(category_table.columns) > 2:  # More than just userid and userfullname
        st.dataframe(category_table, use_container_width=True)
        
        # Download category-specific data
        csv_data = category_table.to_csv(index=False)
        category_name = title.lower().replace(" ", "_")
        st.download_button(
            f"📥 Download {title} (CSV)",
            csv_data,
            f"{category_name}_attributes.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info(f"No {title.lower()} attributes selected or computed.")

if __name__ == "__main__":
    main()