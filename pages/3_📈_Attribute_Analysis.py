import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
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
    
    # Show graph section if attributes have been computed
    if data_manager.get_student_attributes() is not None:
        display_graph_section(data_manager.get_student_attributes())
    
    # Show navigation button if attributes have been computed
    if data_manager.get_student_attributes() is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("🏆 Proceed to Ranking", use_container_width=True, 
                        help="Navigate to the ranking page with computed attributes"):
                st.switch_page("pages/4_🏆_Ranking.py")

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
        
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 6px;
            margin-top: 10px;
            margin-bottom: 15px;
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
    
    with st.spinner("⏳ Computing selected attributes, please wait..."):
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🚀 Activity", 
        "💬 Engagement", 
        "📝 Content", 
        "📋 Exams",
        "📈 Graphs & Visualizations"  # New Graph Tab
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
    
    with tab6:  # New Graph Tab
        display_graph_section(oam_combined)
    
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

def display_graph_section(oam_combined):
    """Display comprehensive graph section for attributes and students"""
    st.header("📈 Attribute & Student Visualizations")
    
    # Check if we have attributes to visualize
    fixed_cols = ["userid", "userfullname"]
    attribute_cols = [col for col in oam_combined.columns if col not in fixed_cols]
    
    if not attribute_cols:
        st.warning("No attributes available for visualization. Please compute attributes first.")
        return
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "📊 Attribute Distribution Analysis",
            "👥 Student Performance Comparison", 
            "🔥 Top Performers by Attribute",
            "📈 Student Attribute Profile",
            "🌐 Correlation Heatmap",
            "📋 Category-wise Analysis"
        ]
    )
    
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
    """Display distribution analysis for individual attributes"""
    st.subheader("📊 Attribute Distribution Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_attribute = st.selectbox(
            "Select Attribute to Analyze",
            attribute_cols,
            key="attr_dist_select"
        )
        
        # Statistics
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
            # Create distribution plot
            fig = px.histogram(
                oam_combined,
                x=selected_attribute,
                title=f"Distribution of {selected_attribute.replace('_', ' ').title()}",
                nbins=20,
                color_discrete_sequence=['#3366CC']
            )
            fig.update_layout(
                xaxis_title=selected_attribute.replace('_', ' ').title(),
                yaxis_title="Number of Students",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot for outlier detection
            fig_box = px.box(
                oam_combined,
                y=selected_attribute,
                title=f"Box Plot - {selected_attribute.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_box, use_container_width=True)

def display_student_comparison(oam_combined, attribute_cols):
    """Display comparison of students across multiple attributes"""
    st.subheader("👥 Student Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_students = st.multiselect(
            "Select Students to Compare",
            options=oam_combined["userfullname"].tolist(),
            default=oam_combined["userfullname"].head(5).tolist()
        )
    
    with col2:
        selected_attributes = st.multiselect(
            "Select Attributes for Comparison",
            options=attribute_cols,
            default=attribute_cols[:5] if len(attribute_cols) >= 3 else attribute_cols
        )
    
    if selected_students and selected_attributes:
        # Filter data for selected students
        comparison_data = oam_combined[oam_combined["userfullname"].isin(selected_students)]
        
        # Create radar chart for comparison
        if len(selected_attributes) >= 3:
            fig_radar = create_radar_chart(comparison_data, selected_students, selected_attributes)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Bar chart comparison
        fig_bar = create_attribute_comparison_bar(comparison_data, selected_students, selected_attributes)
        st.plotly_chart(fig_bar, use_container_width=True)

def create_radar_chart(comparison_data, students, attributes):
    """Create a radar chart for student comparison"""
    fig = go.Figure()
    
    # Normalize data for radar chart
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
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Student Comparison Radar Chart"
    )
    
    return fig

def create_attribute_comparison_bar(comparison_data, students, attributes):
    """Create bar chart comparing students across attributes"""
    # Melt data for plotting
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
    """Display top performers for each attribute"""
    st.subheader("🔥 Top Performers by Attribute")
    
    selected_attribute = st.selectbox(
        "Select Attribute for Ranking",
        attribute_cols,
        key="top_perf_select"
    )
    
    top_n = st.slider("Number of Top Students to Show", 5, 20, 10)
    
    if selected_attribute:
        # Get top performers
        top_students = oam_combined.nlargest(top_n, selected_attribute)[["userfullname", selected_attribute]]
        
        # Create horizontal bar chart
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(top_students, use_container_width=True)

def display_student_profile(oam_combined, attribute_cols):
    """Display individual student attribute profile"""
    st.subheader("📈 Student Attribute Profile")
    
    selected_student = st.selectbox(
        "Select Student",
        options=oam_combined["userfullname"].tolist()
    )
    
    if selected_student:
        student_data = oam_combined[oam_combined["userfullname"] == selected_student].iloc[0]
        
        # Create gauge charts for key metrics
        st.markdown(f"### 📊 Performance Profile: {selected_student}")
        
        # Select top 6 attributes for display
        display_attrs = attribute_cols[:6] if len(attribute_cols) >= 6 else attribute_cols
        
        # Create gauge subplots
        cols = st.columns(3)
        for i, attr in enumerate(display_attrs):
            with cols[i % 3]:
                value = student_data[attr]
                max_val = oam_combined[attr].max()
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = value,
                    title = {'text': attr.replace('_', ' ').title()},
                    gauge = {
                        'axis': {'range': [0, max_val]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_val/3], 'color': "lightgray"},
                            {'range': [max_val/3, 2*max_val/3], 'color': "gray"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
        
        # Overall performance bar chart
        fig_bar = px.bar(
            x=display_attrs,
            y=[student_data[attr] for attr in display_attrs],
            title=f"Attribute Scores - {selected_student}",
            labels={'x': 'Attributes', 'y': 'Score'},
            color=[student_data[attr] for attr in display_attrs],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_correlation_heatmap(oam_combined, attribute_cols):
    """Display correlation heatmap between attributes"""
    st.subheader("🌐 Attribute Correlation Heatmap")
    
    if len(attribute_cols) < 2:
        st.warning("Need at least 2 attributes for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = oam_combined[attribute_cols].corr()
    
    # Create heatmap
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    with st.expander("💡 Correlation Interpretation Guide"):
        st.markdown("""
        **Correlation Values Meaning:**
        - **+1.0**: Perfect positive correlation
        - **+0.7 to +1.0**: Strong positive correlation  
        - **+0.3 to +0.7**: Moderate positive correlation
        - **-0.3 to +0.3**: Weak or no correlation
        - **-0.7 to -0.3**: Moderate negative correlation
        - **-1.0 to -0.7**: Strong negative correlation
        - **-1.0**: Perfect negative correlation
        """)

def display_category_analysis(oam_combined):
    """Display analysis by attribute categories"""
    st.subheader("📋 Category-wise Attribute Analysis")
    
    # Categorize attributes
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
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    
    if not categories:
        st.warning("No categorized attributes available")
        return
    
    selected_category = st.selectbox("Select Category", list(categories.keys()))
    
    if selected_category and categories[selected_category]:
        category_cols = categories[selected_category]
        
        # Calculate category averages
        category_avg = oam_combined[category_cols].mean()
        
        # Create bar chart of category averages
        fig = px.bar(
            x=category_cols,
            y=category_avg.values,
            title=f"{selected_category} Category - Average Scores",
            labels={'x': 'Attributes', 'y': 'Average Score'},
            color=category_avg.values,
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top performers in this category
        st.subheader(f"🏆 Top Performers - {selected_category} Category")
        
        # Calculate category total score
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
            
            st.plotly_chart(fig_top, use_container_width=True)

if __name__ == "__main__":
    main()