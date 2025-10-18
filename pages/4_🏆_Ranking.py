# pages/4_🏆_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    st.title("🏆 Student Ranking")
    st.markdown("Rank students attributes for COCO analysis.")
    
    # Check if attributes are computed
    student_attributes = data_manager.get_student_attributes()
    if student_attributes is None:
        st.warning("📊 Please compute student attributes first on the Attribute Analysis page.")
        return
    
    # Get selected attributes from session state
    if "selected_attributes" not in st.session_state:
        st.warning("❌ No attributes selected for ranking. Please go back to Attribute Analysis page.")
        return
    
    selected_attributes = st.session_state.selected_attributes
    if not selected_attributes:
        st.warning("❌ No attributes selected for ranking. Please select attributes first.")
        return
    
    st.success(f"✅ Ready to rank {len(student_attributes)} students using {len(selected_attributes)} attributes")
    
    # Use Y-value from configuration
    y_value = config.analysis_settings.get('y_value', 1000)
    
    # Simple configuration section
    st.header("⚙️ Ranking Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Y-value display (read-only from configuration)
        st.subheader("🎯 Reference Value (Y)")
        st.info(f"**Y Value:** {y_value}")
        st.caption("💡 Configure Y value in the Configuration page")
        
        # Optional: Allow temporary override for this session
        use_custom_y = st.checkbox("Use custom Y value for this session", value=False)
        if use_custom_y:
            y_value = st.number_input(
                "Custom Y value:",
                min_value=0,
                max_value=100000,
                value=y_value,
                step=100,
                help="Temporary override - won't save to configuration"
            )
    
    with col2:
        st.subheader("📊 Summary")
        st.metric("Students", len(student_attributes))
        st.metric("Attributes", len(selected_attributes))
        st.metric("Y Value", y_value)
    
    # Simple ranking directions
    st.subheader("📈 Ranking Directions")
    
    # Create attribute directions
    attr_directions = {}
    direction_info = []
    
    for attr in selected_attributes:
        # Set direction: 0 = Lower is better, 1 = Higher is better
        if attr.startswith("deadline_exceeded_posts_"):
            direction = 0  # Lower is better for deadline exceeded
            direction_text = "🔻 Lower is better"
        elif attr == "avg_reply_time":
            direction = 0  # Lower is better for reply time
            direction_text = "🔻 Lower is better"
        else:
            direction = 1  # Higher is better for most attributes
            direction_text = "🔺 Higher is better"
        
        attr_directions[attr] = direction
        direction_info.append(f"**{attr.replace('_', ' ').title()}:** {direction_text}")
    
    # Show directions in a compact format
    with st.expander("View Ranking Directions", expanded=False):
        for info in direction_info:
            st.write(info)
    
    # Run ranking
    st.markdown("---")
    if st.button("🚀 Run Student Ranking", use_container_width=True, type="primary"):
        ranked_df = rank_students(student_attributes, selected_attributes, attr_directions, y_value)
        
        if ranked_df is not None:
            # Store results
            data_manager.store_ranked_results(ranked_df)
            
            # Display results
            display_ranking_results(ranked_df, selected_attributes, data_manager, y_value)
    
    # Show navigation button if ranking has been computed
    if data_manager.get_ranked_results() is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("🔍 Proceed to COCO Analysis", use_container_width=True, 
                        help="Navigate to COCO Analysis page with ranked data"):
                st.switch_page("pages/5_🔍_COCO_Analysis.py")

def rank_students(df_oam, selected_attrs, attr_directions, y_value):
    """Rank students based on selected attributes and directions"""
    try:
        rdf = df_oam.copy()
        
        # Filter only selected attributes that exist in the dataframe
        valid_attrs = [attr for attr in selected_attrs if attr in rdf.columns]
        
        if not valid_attrs:
            st.error("❌ No valid attributes found for ranking.")
            return None
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("Ranking students...")
        
        for i, attr in enumerate(valid_attrs):
            progress_bar.progress(int((i / len(valid_attrs)) * 100))
            
            direction = attr_directions.get(attr, 1)
            ascending = True if direction == 0 else False
            
            # Handle NaN values by filling with worst possible value for ranking
            if ascending:  # Lower is better, so NaN should get worst rank (high number)
                fill_value = rdf[attr].max() if not rdf[attr].isna().all() else 1
            else:  # Higher is better, so NaN should get worst rank (low number)
                fill_value = rdf[attr].min() if not rdf[attr].isna().all() else 0
                
            temp_series = rdf[attr].fillna(fill_value)
            rdf[attr + "_rank"] = temp_series.rank(method="min", ascending=ascending).astype(int)
        
        progress_bar.progress(100)
        progress_text.text("Ranking complete!")
        
        # Add Y value column (last row gets special value for COCO)
        rdf["Y_value"] = y_value
        if len(rdf) > 0:
            rdf.loc[rdf.index[-1], "Y_value"] = 100000
        
        st.success(f"✅ Successfully ranked {len(rdf)} students!")
        return rdf
        
    except Exception as e:
        st.error(f"❌ Error during ranking: {e}")
        return None

def display_ranking_results(ranked_df, selected_attributes, data_manager, y_value):
    """Display ranking results in a simple table"""
    
    # Create rank columns list
    rank_columns = [attr + "_rank" for attr in selected_attributes if attr + "_rank" in ranked_df.columns]
    
    if not rank_columns:
        st.error("No ranking columns found.")
        return
    
    st.header("📊 Ranking Results")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(ranked_df))
    with col2:
        avg_rank = ranked_df[rank_columns].mean().mean()
        st.metric("Average Rank", f"{avg_rank:.1f}")
    with col3:
        st.metric("Y Value Used", y_value)
    
    # Calculate overall ranking
    ranked_df["Average_Rank"] = ranked_df[rank_columns].mean(axis=1)
    ranked_df["Overall_Rank"] = ranked_df["Average_Rank"].rank(method="min").astype(int)
    
    # Create display columns - only show ranks, not original values
    display_columns = ["userfullname", "Overall_Rank", "Average_Rank"] + rank_columns
    
    # Sort by overall rank
    display_df = ranked_df[display_columns].sort_values("Overall_Rank")
    
    # Rename columns for better display
    column_renames = {
        "userfullname": "Student Name",
        "Overall_Rank": "Overall Rank", 
        "Average_Rank": "Average Rank"
    }
    
    for col in rank_columns:
        friendly_name = col.replace("_rank", "").replace("_", " ").title()
        column_renames[col] = friendly_name
    
    display_df = display_df.rename(columns=column_renames)
    
    # Display the ranked table
    st.subheader("🏆 Ranked Students")
    st.dataframe(display_df, use_container_width=True)
    
    # Show top performers
    st.subheader("🎯 Top Performers")
    top_5 = display_df.head(5)
    for idx, row in top_5.iterrows():
        rank_icon = "🥇" if row["Overall Rank"] == 1 else "🥈" if row["Overall Rank"] == 2 else "🥉" if row["Overall Rank"] == 3 else "🏅"
        st.write(f"{rank_icon} **{row['Overall Rank']}.** {row['Student Name']} (Avg Rank: {row['Average Rank']:.1f})")
    
    # Export options
    st.markdown("---")
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download ranked data
        csv_data = ranked_df.to_csv(index=False)
        st.download_button(
            "📥 Download Ranking Data (CSV)",
            csv_data,
            f"student_ranking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()