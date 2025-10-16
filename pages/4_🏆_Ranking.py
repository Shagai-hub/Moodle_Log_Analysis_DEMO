# pages/4_🏆_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
import plotly.express as px
import plotly.graph_objects as go

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    st.title("🏆 Student Ranking")
    st.markdown("Rank students based on their engagement attributes and prepare for COCO analysis.")
    
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
    
    # Ranking configuration
    st.header("⚙️ Ranking Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Y-value configuration
        st.subheader("🎯 Reference Value (Y)")
        st.markdown("The Y-value is used as a reference point in COCO analysis.")
        y_value = st.number_input(
            "Set Y value for analysis:",
            min_value=0,
            max_value=100000,
            value=config.analysis_settings.get('y_value', 1000),
            step=100,
            help="Typically set to 1000 for normal analysis"
        )
    
    with col2:
        st.subheader("📊 Summary")
        st.metric("Students", len(student_attributes))
        st.metric("Attributes", len(selected_attributes))
        st.metric("Y Value", y_value)
    
    # Ranking directions with enhanced UI
    st.subheader("📈 Ranking Directions")
    st.markdown("Configure whether higher or lower values are better for each attribute")
    
    # Create attribute directions
    attr_directions = create_attribute_directions(selected_attributes)
    
    # Display directions in an interactive table
    display_ranking_directions(selected_attributes, attr_directions)
    
    # Run ranking
    st.markdown("---")
    if st.button("🚀 Run Student Ranking", use_container_width=True, type="primary"):
        with st.spinner("Ranking students..."):
            ranked_df = rank_students(student_attributes, selected_attributes, attr_directions, y_value)
            
            if ranked_df is not None:
                # Store results
                data_manager.store_ranked_results(ranked_df)
                
                # Display results
                display_ranking_results(ranked_df, selected_attributes, data_manager)
    
    # Show previous results if available
    existing_ranked = data_manager.get_ranked_results()
    if existing_ranked is not None:
        st.markdown("---")
        st.subheader("📋 Previous Ranking Results")
        with st.expander("Show Previous Ranking", expanded=False):
            display_ranking_results(existing_ranked, selected_attributes, data_manager, is_previous=True)

def create_attribute_directions(selected_attributes):
    """Create ranking directions for attributes"""
    attr_directions = {}
    
    for attr in selected_attributes:
        # Set direction: 0 = Lower is better, 1 = Higher is better
        if attr.startswith("deadline_exceeded_posts_"):
            direction = 0  # Lower is better for deadline exceeded
        elif attr == "avg_reply_time":
            direction = 0  # Lower is better for reply time
        else:
            direction = 1  # Higher is better for most attributes
        
        attr_directions[attr] = direction
    
    return attr_directions

def display_ranking_directions(selected_attributes, attr_directions):
    """Display ranking directions in an interactive format"""
    
    # Create a DataFrame for display
    direction_data = []
    for attr in selected_attributes:
        direction = attr_directions.get(attr, 1)
        direction_text = "🔻 Lower is better" if direction == 0 else "🔺 Higher is better"
        direction_icon = "📉" if direction == 0 else "📈"
        
        direction_data.append({
            "Attribute": attr.replace("_", " ").title(),
            "Direction": direction_text,
            "Icon": direction_icon,
            "Preference": "Minimize" if direction == 0 else "Maximize"
        })
    
    dir_df = pd.DataFrame(direction_data)
    
    # Display as styled table
    st.dataframe(
        dir_df,
        use_container_width=True,
        column_config={
            "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
            "Direction": st.column_config.TextColumn("Ranking Direction", width="medium"),
            "Icon": st.column_config.TextColumn("", width="small"),
            "Preference": st.column_config.TextColumn("Goal", width="small")
        },
        hide_index=True
    )
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    maximize_count = sum(1 for d in attr_directions.values() if d == 1)
    minimize_count = sum(1 for d in attr_directions.values() if d == 0)
    
    with col1:
        st.metric("Maximize Attributes", maximize_count)
    with col2:
        st.metric("Minimize Attributes", minimize_count)
    with col3:
        total_impact = maximize_count - minimize_count
        st.metric("Overall Bias", "Positive" if total_impact > 0 else "Negative")

def rank_students(df_oam, selected_attrs, attr_directions, y_value):
    """Rank students based on selected attributes and directions"""
    try:
        rdf = df_oam.copy()
        
        # Filter only selected attributes that exist in the dataframe
        valid_attrs = [attr for attr in selected_attrs if attr in rdf.columns]
        
        if not valid_attrs:
            st.error("❌ No valid attributes found for ranking.")
            return None
        
        st.info(f"📊 Ranking using {len(valid_attrs)} attributes...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        
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
        
        # Add Y value column (last row gets special value for COCO)
        rdf["Y_value"] = y_value
        if len(rdf) > 0:
            rdf.loc[rdf.index[-1], "Y_value"] = 100000
        
        st.success(f"✅ Successfully ranked {len(rdf)} students!")
        return rdf
        
    except Exception as e:
        st.error(f"❌ Error during ranking: {e}")
        return None

def display_ranking_results(ranked_df, selected_attributes, data_manager, is_previous=False):
    """Display ranking results with visualizations"""
    
    # Create rank columns list
    rank_columns = [attr + "_rank" for attr in selected_attributes if attr + "_rank" in ranked_df.columns]
    
    if not rank_columns:
        st.error("No ranking columns found.")
        return
    
    st.header("📊 Ranking Results")
    
    # Summary statistics
    st.subheader("📈 Ranking Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(ranked_df))
    with col2:
        avg_rank = ranked_df[rank_columns].mean().mean()
        st.metric("Average Rank", f"{avg_rank:.1f}")
    with col3:
        min_rank = ranked_df[rank_columns].min().min()
        st.metric("Best Rank", int(min_rank))
    with col4:
        max_rank = ranked_df[rank_columns].max().max()
        st.metric("Worst Rank", int(max_rank))
    
    # Tabbed interface for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Overall Ranking", "📋 Detailed Ranks", "📊 Visualizations", "💾 Export"])
    
    with tab1:
        display_overall_ranking(ranked_df, rank_columns)
    
    with tab2:
        display_detailed_ranks(ranked_df, selected_attributes)
    
    with tab3:
        display_ranking_visualizations(ranked_df, rank_columns)
    
    with tab4:
        display_export_options(ranked_df, data_manager, is_previous)

def display_overall_ranking(ranked_df, rank_columns):
    """Display overall ranking based on average rank"""
    
    # Calculate average rank across all attributes
    ranked_df["Average_Rank"] = ranked_df[rank_columns].mean(axis=1)
    ranked_df["Overall_Rank"] = ranked_df["Average_Rank"].rank(method="min").astype(int)
    
    # Sort by overall rank
    overall_ranked = ranked_df.sort_values("Overall_Rank")[["userfullname", "Overall_Rank", "Average_Rank"] + rank_columns]
    
    st.subheader("🏆 Overall Student Ranking")
    st.markdown("Students ranked by average performance across all attributes")
    
    # Display top 10 and bottom 10
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥇 Top Performers**")
        top_10 = overall_ranked.head(10)
        for idx, row in top_10.iterrows():
            rank_icon = "🥇" if row["Overall_Rank"] == 1 else "🥈" if row["Overall_Rank"] == 2 else "🥉" if row["Overall_Rank"] == 3 else "🏅"
            st.write(f"{rank_icon} **{row['Overall_Rank']}.** {row['userfullname']} (Avg: {row['Average_Rank']:.1f})")
    
    with col2:
        st.markdown("**📊 Need Improvement**")
        bottom_10 = overall_ranked.tail(10).iloc[::-1]  # Reverse to show worst first
        for idx, row in bottom_10.iterrows():
            st.write(f"🔻 **{row['Overall_Rank']}.** {row['userfullname']} (Avg: {row['Average_Rank']:.1f})")
    
    # Full ranking table
    with st.expander("📋 View Complete Ranking Table", expanded=False):
        st.dataframe(overall_ranked, use_container_width=True)

def display_detailed_ranks(ranked_df, selected_attributes):
    """Display detailed ranking per attribute"""
    
    st.subheader("📋 Detailed Attribute Rankings")
    st.markdown("See how students rank for each individual attribute")
    
    # Create a clean display dataframe
    display_cols = ["userfullname"]
    for attr in selected_attributes:
        rank_col = attr + "_rank"
        if rank_col in ranked_df.columns:
            display_cols.extend([attr, rank_col])
    
    if len(display_cols) > 1:
        detailed_df = ranked_df[display_cols].copy()
        
        # Rename columns for better display
        column_renames = {}
        for col in detailed_df.columns:
            if col == "userfullname":
                column_renames[col] = "Student"
            elif col.endswith("_rank"):
                column_renames[col] = col.replace("_rank", "").replace("_", " ").title() + " Rank"
            else:
                column_renames[col] = col.replace("_", " ").title()
        
        detailed_df = detailed_df.rename(columns=column_renames)
        st.dataframe(detailed_df, use_container_width=True)
    else:
        st.info("No detailed ranking data available.")

def display_ranking_visualizations(ranked_df, rank_columns):
    """Display visualizations of ranking results"""
    
    st.subheader("📊 Ranking Visualizations")
    
    if len(rank_columns) == 0:
        st.info("No ranking data available for visualization.")
        return
    
    # Calculate average ranks for visualization
    avg_ranks = ranked_df[rank_columns].mean()
    attribute_names = [col.replace("_rank", "").replace("_", " ").title() for col in rank_columns]
    
    # Create bar chart of average ranks per attribute
    fig = px.bar(
        x=attribute_names,
        y=avg_ranks.values,
        title="📈 Average Student Ranks by Attribute",
        labels={"x": "Attribute", "y": "Average Rank"},
        color=avg_ranks.values,
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Rank distribution heatmap
    st.markdown("**🎯 Rank Distribution Heatmap**")
    
    # Sample of students for heatmap (limit to 20 for readability)
    heatmap_data = ranked_df.set_index("userfullname")[rank_columns].head(20)
    
    if len(heatmap_data) > 0:
        fig_heatmap = px.imshow(
            heatmap_data,
            aspect="auto",
            title="Student Rank Heatmap (Top 20 Students)",
            color_continuous_scale="RdYlGn_r",  # Reversed so green=good, red=bad
            labels=dict(color="Rank")
        )
        fig_heatmap.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough data for heatmap visualization.")

def display_export_options(ranked_df, data_manager, is_previous=False):
    """Display export options for ranking results"""
    
    st.subheader("💾 Export Ranking Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export current ranking
        if not is_previous:
            csv_data = ranked_df.to_csv(index=False)
            st.download_button(
                "📥 Download Current Ranking (CSV)",
                csv_data,
                f"student_ranking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("This is a previous ranking. Compute new ranking to download.")
    
    with col2:
        # Proceed to COCO analysis
        if not is_previous:
            if st.button("🔍 Proceed to COCO Analysis", use_container_width=True):
                st.session_state.proceed_to_coco = True
                st.rerun()
        else:
            st.button("🔄 Compute New Ranking", use_container_width=True)
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(ranked_df.head(10), use_container_width=True)
        st.write(f"Full dataset: {ranked_df.shape[0]} rows × {ranked_df.shape[1]} columns")

if __name__ == "__main__":
    main()