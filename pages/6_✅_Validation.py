# pages/5_✅_Validation.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.coco_utils import send_coco_request, parse_coco_html, invert_ranking, prepare_coco_matrix
from assets.ui_components import apply_theme, divider, info_panel, nav_footer, page_header, section_header

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

apply_theme()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    page_header(
        "Validation Dashboard",
        "Validate COCO results by comparing original and inverted rankings.",
        icon="🎯",
        align="left",
        compact=True,
    )
    
    # Check if required data is available
    ranked_data = data_manager.get_ranked_results()
    coco_results = data_manager.get_coco_results()
    
    if ranked_data is None or coco_results is None:
        st.error("📊 **Data Required**", icon="🚨")
        info_panel("Please complete the COCO analysis first to enable validation.", icon="ℹ️")
        divider()
        nav_footer(
            back={
                "label": "⬅️ Back to COCO Analysis",
                "page": "pages/5_🔍_COCO_Analysis.py",
                "key": "nav_back_to_coco_missing_validation",
                "fallback": "🔍 COCO Analysis",
            },
            message="Complete the COCO analysis before running validation.",
        )
        return
   
    # Success banner with simplified stats
    info_panel(
        f"<strong>Students ready for validation:</strong> {len(ranked_data)}",
        icon="📈",
    )
    
    st.write("")
    # Configuration and preview section
    with st.expander("🔧 **Data Preview**", expanded=False):
        if 'table_4' in coco_results:
            main_table = coco_results['table_4']
            
            # Simple table preview
            st.dataframe(
                main_table.head(8), 
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing 8 of {main_table.shape[0]} rows × {main_table.shape[1]} columns")
            
            # Basic column validation
            delta_found = 'Delta/Tény' in main_table.columns
            becsl_found = 'Becslés' in main_table.columns
            
            if delta_found and becsl_found:
                st.success("✅ **Validation Ready** - All required columns are available!", icon="🎯")
            else:
                st.error("❌ **Missing Required Columns**", icon="🚨")
                if not delta_found:
                    st.error("Missing 'Delta/Tény' column")
                if not becsl_found:
                    st.error("Missing 'Becslés' column")
        else:
            st.error("📋 **Table Missing** - 'table_4' not found in COCO results", icon="❌")
    
    # Run Validation Button - OUTSIDE expander and prominent
    divider()
    
    # Check if we can run validation
    can_run_validation = ('table_4' in coco_results and 
                         'Delta/Tény' in coco_results['table_4'].columns and 
                         'Becslés' in coco_results['table_4'].columns)
    
    if can_run_validation:
        if st.button(
            "🚀 Run Comprehensive Validation", 
            type="primary", 
            use_container_width=True,
            help="Validate COCO results by comparing with inverted rankings",
            key="pulse"
        ):
            run_validation_analysis(ranked_data, coco_results, data_manager)
    else:
        st.button(
            "🚀 Run Comprehensive Validation", 
            type="secondary", 
            use_container_width=True,
            disabled=True,
            help="Cannot run validation - missing required data",
            key="pulse"
        )
        st.warning("Please ensure COCO analysis completed successfully with required columns.")
    
    # Display validation results if available - OUTSIDE expander and full width
    validation_results = data_manager.get_validation_results()
    if validation_results is not None:
        display_validation_results(validation_results, data_manager)
    
    forward_spec = {
        "label": "🤖 Go to AI Insights",
        "page": "pages/7_🤖_AI_Insights.py",
        "key": "nav_forward_to_ai_insights",
        "fallback": "🤖 AI Insights",
    }

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to COCO Analysis",
            "page": "pages/5_🔍_COCO_Analysis.py",
            "key": "nav_back_to_coco_footer",
            "fallback": "🔍 COCO Analysis",
        },
        forward=forward_spec,
    )

def run_validation_analysis(ranked_data, coco_results, data_manager):
    """Run the complete validation analysis with modern progress tracking"""
    
    # Modern progress container
    with st.status("🔄 **Running Validation Analysis...**", expanded=True) as status:
        try:
            # Get table_4
            main_table = coco_results['table_4']
            
            # Step 1: Prepare inverted matrix
            st.write("📊 Preparing inverted matrix...")
            matrix_df = ranked_data.drop(columns=["userid", "userfullname"], errors='ignore')
            inverted_matrix_df = invert_ranking(matrix_df)
            
            # Step 2: Convert to COCO format
            st.write("🔄 Converting to COCO format...")
            inverted_matrix_data = prepare_coco_matrix(inverted_matrix_df)
            
            # Step 3: Send to COCO
            st.write("🌐 Sending inverted matrix to COCO...")
            stair_value = len(ranked_data)
            resp = send_coco_request(
                matrix_data=inverted_matrix_data,
                job_name="StudentRankingInverted",
                stair=str(stair_value),
                timeout=180
            )
            
            # Step 4: Parse inverted results
            st.write("📈 Parsing inverted COCO results...")
            inverted_tables = parse_coco_html(resp)
            
            if not inverted_tables:
                st.error("❌ No results received from inverted COCO analysis")
                return
            
            # Step 5: Perform validation
            st.write("✅ Performing validation analysis...")
            if 'table_4' in inverted_tables:
                inverted_main_table = inverted_tables['table_4']
                validation_results = perform_validation(
                    main_table, inverted_main_table, ranked_data
                )
                
                if validation_results is not None and not validation_results.empty:
                    data_manager.store_validation_results(validation_results)
                    status.update(label="✅ **Validation Complete!**", state="complete", expanded=False)
                else:
                    status.update(label="❌ **Validation Failed**", state="error")
                    st.error("Validation failed to produce meaningful results")
            else:
                status.update(label="❌ **Table Missing**", state="error")
                st.error("Required table_4 not found in inverted results")
                
        except Exception as e:
            status.update(label="❌ **Validation Error**", state="error")
            st.error(f"Validation process failed: {str(e)}")
            import traceback
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())

def perform_validation(original_table, inverted_table, ranked_data):
    """Validate COCO analysis results by comparing original and inverted deltas."""
    try:
        delta_col = 'Delta/Tény'
        becsl_col = 'Becslés'
        
        # Check if columns exist
        if delta_col not in original_table.columns or delta_col not in inverted_table.columns:
            st.error(f"❌ Delta column '{delta_col}' not found")
            return None
            
        if becsl_col not in original_table.columns:
            st.error(f"❌ Estimation column '{becsl_col}' not found") 
            return None
        
        # Convert to numeric and handle invalid values
        original_delta = pd.to_numeric(original_table[delta_col], errors='coerce')
        inverted_delta = pd.to_numeric(inverted_table[delta_col], errors='coerce')
        original_becsl = pd.to_numeric(original_table[becsl_col], errors='coerce')
        
        # Filter out rows with NaN values
        valid_indices = original_delta.notna() & inverted_delta.notna()
        original_delta = original_delta[valid_indices]
        inverted_delta = inverted_delta[valid_indices]
        ranked_data = ranked_data.loc[valid_indices]
        original_becsl = original_becsl[valid_indices]
        
        # Calculate validation: original_delta * inverted_delta <= 0 is valid
        validation_product = original_delta * inverted_delta
        is_valid = validation_product <= 0
        
        # Create validation results
        validation_results = ranked_data.copy()
        validation_results['Becslés'] = original_becsl
        validation_results['Original_Delta'] = original_delta
        validation_results['Inverted_Delta'] = inverted_delta
        validation_results['Validation_Product'] = validation_product
        validation_results['Validation_Result'] = is_valid.map({True: 'Valid', False: 'Invalid'})
        validation_results['Is_Valid'] = is_valid
        
        # Add ranking based on Becsl_s score
        validation_results['Final_Rank'] = validation_results['Becslés'].rank(ascending=False, method='min').astype(int)
        
        return validation_results
        
    except Exception as e:
        st.error(f"❌ Validation calculation error: {str(e)}")
        return None

def display_validation_results(validation_results, data_manager):
    """Display modern validation results with enhanced visuals - FULL SCREEN WIDTH"""
    
    section_header("Validation Results Dashboard", icon="📊")
    
    # Enhanced metrics with visual appeal
    valid_count = validation_results['Is_Valid'].sum()
    total_count = len(validation_results)
    validity_percentage = (valid_count / total_count) * 100
    
    # Create metrics with visual indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Students", 
            total_count,
            help="Number of students included in validation"
        )
    
    with col2:
        st.metric(
            "Valid Results", 
            f"{valid_count}",
            delta=f"{validity_percentage:.1f}% success rate"
        )
    
    with col3:
        # Color-coded validity rate
        if validity_percentage >= 80:
            icon = "✅"
            delta_color = "normal"
        elif validity_percentage >= 50:
            icon = "⚠️"
            delta_color = "off"
        else:
            icon = "❌"
            delta_color = "inverse"
            
        st.metric(
            "Validity Rate", 
            f"{validity_percentage:.1f}%",
            delta=icon,
            delta_color=delta_color
        )
    
    
    # Visualization section with tabs - USING FULL WIDTH
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Score Distribution", 
        "🔍 Validation Analysis", 
        "📋 Detailed Results", 
        "⚠️ Review Cases"
    ])
    
    with tab1:
        # Enhanced score distribution visualization
        section_header("Final Score Distribution by Validation Status", tight=True)
        
        # Create interactive Plotly histogram
        fig = px.histogram(
            validation_results, 
            x='Becslés',
            color='Validation_Result',
            nbins=20,
            color_discrete_map={'Valid': '#00CC96', 'Invalid': '#EF553B'},
            opacity=0.7,
            barmode='overlay'
        )
        
        fig.update_layout(
            title="Distribution of Final Scores",
            xaxis_title="Final Score (Becslés)",
            yaxis_title="Number of Students",
            legend_title="Validation Result",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional statistics
    
    with tab2:
        # Scatter plot for delta comparison
        section_header("Delta Comparison: Original vs Inverted", tight=True)
        
        scatter_fig = px.scatter(
            validation_results,
            x='Original_Delta',
            y='Inverted_Delta',
            color='Validation_Result',
            color_discrete_map={'Valid': '#00CC96', 'Invalid': '#EF553B'},
            hover_data=['userfullname', 'Final_Rank', 'Becslés'],
            size='Becslés',
            size_max=15,
            opacity=0.7
        )
        
        # Add quadrant lines
        scatter_fig.add_hline(y=0, line_dash="dash", line_color="grey")
        scatter_fig.add_vline(x=0, line_dash="dash", line_color="grey")
        
        scatter_fig.update_layout(
            title="Original Delta vs Inverted Delta",
            xaxis_title="Original Delta",
            yaxis_title="Inverted Delta",
            legend_title="Validation Result"
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Validation insights
        section_header("Validation Insights", icon="📊", tight=True)
        
        insights_col1, insights_col2 = st.columns(2)

        if validity_percentage >= 80:
            status_icon = "✅"
            status_message = "**Strong Consistency** — results show high reliability"
        elif validity_percentage >= 50:
            status_icon = "⚠️"
            status_message = "**Moderate Consistency** — some results need review"
        else:
            status_icon = "❌"
            status_message = "**Low Consistency** — significant review required"

        with insights_col1:
            info_panel(
                f"{status_message}<br><br>"
                f"<strong>Success Rate</strong>: {validity_percentage:.1f}%<br>"
                f"<strong>Valid Cases</strong>: {valid_count}/{total_count}",
                icon=status_icon,
            )

        delta_correlation = validation_results['Original_Delta'].corr(validation_results['Inverted_Delta'])
        correlation_hint = (
            "Low correlation expected for valid inversion"
            if abs(delta_correlation) < 0.3
            else "Unexpected correlation pattern detected"
        )

        with insights_col2:
            info_panel(
                f"<strong>Delta Correlation</strong>: {delta_correlation:.3f}<br>"
                f"{correlation_hint}<br><br>"
                f"<strong>Mean Delta (Original)</strong>: {validation_results['Original_Delta'].mean():.3f}<br>"
                f"<strong>Mean Delta (Inverted)</strong>: {validation_results['Inverted_Delta'].mean():.3f}",
                icon="📐",
            )

    with tab3:
        # Enhanced results table - FULL WIDTH
        section_header("Detailed Validation Results", icon="📋", tight=True)
        
        display_columns = ["userfullname", "Final_Rank", "Becslés", "Validation_Result", "Original_Delta", "Inverted_Delta"]
        display_df = validation_results[display_columns].copy()
        display_df = display_df.rename(columns={
            "userfullname": "Student Name",
            "Final_Rank": "Rank",
            "Becslés": "Final Score",
            "Validation_Result": "Validation",
            "Original_Delta": "Original Delta",
            "Inverted_Delta": "Inverted Delta"
        })
        
        # Sort by rank and add styling
        display_df = display_df.sort_values("Rank")
        
        # Add color formatting for validation column
        def color_validation(val):
            color = 'color: green' if val == 'Valid' else 'color: red'
            return color
        
        styled_df = display_df.style.applymap(
            color_validation, 
            subset=['Validation']
        ).format({
            "Final Score": "{:.3f}",
            "Original Delta": "{:.3f}",
            "Inverted Delta": "{:.3f}"
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download option
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Detailed Results (CSV)",
            csv_data,
            f"validation_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with tab4:
        # Enhanced invalid cases analysis
        invalid_cases = validation_results[~validation_results['Is_Valid']]
        
        if not invalid_cases.empty:
            section_header("Cases Requiring Review", icon="⚠️", tight=True)
            
            # Invalid cases metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Invalid Cases", len(invalid_cases))
            with col2:
                st.metric("Percentage of Total", f"{(len(invalid_cases)/total_count)*100:.1f}%")
            with col3:
                avg_invalid_score = invalid_cases['Becslés'].mean()
                st.metric("Avg Score (Invalid)", f"{avg_invalid_score:.3f}")
            
            # Invalid cases table
            invalid_display = invalid_cases[[
                "userfullname", "Final_Rank", "Becslés", 
                "Original_Delta", "Inverted_Delta", "Validation_Product"
            ]].copy()
            
            invalid_display = invalid_display.rename(columns={
                "userfullname": "Student Name",
                "Final_Rank": "Rank",
                "Becslés": "Final Score",
                "Original_Delta": "Original Delta", 
                "Inverted_Delta": "Inverted Delta",
                "Validation_Product": "Product"
            }).sort_values("Rank")
            
            # Format numeric columns
            invalid_display = invalid_display.round({
                "Final Score": 3,
                "Original Delta": 3,
                "Inverted Delta": 3,
                "Product": 3
            })
            
            st.dataframe(invalid_display, use_container_width=True)
            
            # Analysis of invalid cases
            with st.expander("🔍 **Invalid Cases Analysis**"):
                st.write("**Pattern Analysis:**")
                
                # Check if invalid cases cluster in certain score ranges
                score_bins = pd.cut(invalid_cases['Becslés'], bins=5)
                bin_counts = score_bins.value_counts().sort_index()
                
                st.write("**Score Distribution of Invalid Cases:**")
                for bin_range, count in bin_counts.items():
                    st.write(f"• {bin_range}: {count} cases")
                
                # Delta analysis
                st.write("**Delta Analysis:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Original Delta Range: {invalid_cases['Original_Delta'].min():.3f} to {invalid_cases['Original_Delta'].max():.3f}")
                with col2:
                    st.write(f"Inverted Delta Range: {invalid_cases['Inverted_Delta'].min():.3f} to {invalid_cases['Inverted_Delta'].max():.3f}")
        
        else:
            st.success("🎉 **Excellent!** No invalid cases found requiring review.")
    
    # Final export section
    divider()
    
    chart_df = validation_results[["userfullname", "Becslés", "Validation_Result", "Final_Rank"]].copy()
    chart_df = chart_df.rename(columns={
        "userfullname": "Student",
        "Becslés": "Score",
        "Validation_Result": "Validation",
        "Final_Rank": "Rank"
    })
    
    # Create interactive bar chart
    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Student:N", sort=None, title="Student", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Score:Q", title="Final Score (Becsl_s)"),
            color=alt.Color("Validation:N",
                          scale=alt.Scale(domain=["Valid", "Invalid"],
                                        range=["#00ff00", "#ff0000"]),
                          legend=alt.Legend(title="Validation Result")),
            tooltip=["Student", "Score", "Validation", "Rank"]
        )
        .properties(
            width=700,
            height=400,
            title="Student Final Scores with Validation Results"
        )
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    section_header("Export Options", icon="💾", tight=True)
    
    
    summary_data = {
        'Metric': ['Total Students', 'Valid Cases', 'Invalid Cases', 'Validity Rate', 'Average Score'],
        'Value': [
            total_count, 
            valid_count, 
            len(validation_results[~validation_results['Is_Valid']]),
            f"{validity_percentage:.1f}%",
            f"{validation_results['Becslés'].mean():.3f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False)
    
    st.download_button(
        "📄 Download Summary Report (CSV)",
        summary_csv,
        f"validation_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
        help="High-level validation metrics and summary"
        )


if __name__ == "__main__":
    main()
