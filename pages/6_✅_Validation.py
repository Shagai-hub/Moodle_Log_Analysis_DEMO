# pages/5_✅_Validation.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.coco_utils import send_coco_request, parse_coco_html, invert_ranking, prepare_coco_matrix

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    # Modern header with icon and description
    st.header("🎯 Validation Dashboard", divider="rainbow")
    st.markdown("Validate COCO analysis results by comparing original and inverted rankings for accuracy assessment.")
    
    # Check if required data is available
    ranked_data = data_manager.get_ranked_results()
    coco_results = data_manager.get_coco_results()
    
    if ranked_data is None or coco_results is None:
        st.error("📊 **Data Required**", icon="🚨")
        with st.container(border=True):
            st.markdown("Please complete the COCO analysis first to enable validation.")
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🏃 Go to COCO Analysis", use_container_width=True):
                    st.switch_page("pages/4_📊_COCO_Analysis.py")
        return
    
    # Success banner with stats
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Students Ready", len(ranked_data), border=True)
        with col2:
            st.metric("🔢 Data Points", f"{len(ranked_data) * (len(ranked_data.columns)-2):,}", border=True)
        with col3:
            st.metric("📋 Tables Available", len(coco_results), border=True)
    
    # Configuration and preview section
    with st.expander("🔧 **Configuration & Data Preview**", expanded=True):
        if 'table_4' in coco_results:
            main_table = coco_results['table_4']
            
            # Enhanced table preview with tabs
            tab1, tab2 = st.tabs(["📊 Table Preview", "🔍 Column Analysis"])
            
            with tab1:
                st.dataframe(
                    main_table.head(8), 
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing 8 of {main_table.shape[0]} rows × {main_table.shape[1]} columns")
            
            with tab2:
                # Column validation with visual indicators
                delta_found = 'Delta/Tény' in main_table.columns
                becsl_found = 'Becslés' in main_table.columns
                
                col1, col2 = st.columns(2)
                with col1:
                    if delta_found:
                        st.success("✅ **Delta Column**", icon="✅")
                        st.metric("Delta Values", f"{len(main_table['Delta/Tény'])}", delta="Ready")
                    else:
                        st.error("❌ **Missing Delta Column**", icon="❌")
                        
                with col2:
                    if becsl_found:
                        st.success("✅ **Estimation Column**", icon="✅")
                        st.metric("Estimation Values", f"{len(main_table['Becslés'])}", delta="Ready")
                    else:
                        st.error("❌ **Missing Estimation Column**", icon="❌")
                
                # Show all available columns
                st.write("**All Available Columns:**")
                cols_per_row = 4
                columns = list(main_table.columns)
                for i in range(0, len(columns), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(columns):
                            with col:
                                st.code(columns[i + j], language="text")
            
            # Validation readiness check
            if delta_found and becsl_found:
                st.success("🎯 **Validation Ready** - All required columns are available!", icon="🎯")
                
                # Modern button with icon
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col2:
                    if st.button(
                        "🚀 Run Comprehensive Validation", 
                        type="primary", 
                        use_container_width=True,
                        help="Validate COCO results by comparing with inverted rankings"
                    ):
                        run_validation_analysis(ranked_data, coco_results, data_manager)
            else:
                st.error("🔧 **Configuration Required** - Please check your COCO analysis results.", icon="🚨")
                
        else:
            st.error("📋 **Table Missing** - 'table_4' not found in COCO results", icon="❌")
            with st.container(border=True):
                st.write("Available tables in results:")
                for table_name in coco_results.keys():
                    st.write(f"• {table_name}")
    
    # Navigation footer
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to COCO Analysis", use_container_width=True):
            st.switch_page("pages/4_📊_COCO_Analysis.py")
    with col2:
        if st.button("🤖 Get AI Insights ➡️", use_container_width=True):
            st.switch_page("pages/7_🤖_AI_Insights.py")

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
                    
                    # Display results
                    display_validation_results(validation_results, data_manager)
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
    """Display modern validation results with enhanced visuals"""
    
    st.header("📊 Validation Results Dashboard", divider="rainbow")
    
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
    
    with col4:
        # Overall status with emoji
        if validity_percentage >= 80:
            status = "EXCELLENT ✅"
            status_color = "green"
        elif validity_percentage >= 50:
            status = "GOOD ⚠️"
            status_color = "orange"
        else:
            status = "NEEDS REVIEW ❌"
            status_color = "red"
            
        st.metric(
            "Overall Status", 
            status
        )
    
    # Visualization section with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Score Distribution", 
        "🔍 Validation Analysis", 
        "📋 Detailed Results", 
        "⚠️ Review Cases"
    ])
    
    with tab1:
        # Enhanced score distribution visualization
        st.subheader("Final Score Distribution by Validation Status")
        
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
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_score_valid = validation_results[validation_results['Is_Valid']]['Becslés'].mean()
            st.metric("Avg Score (Valid)", f"{avg_score_valid:.3f}")
        with col2:
            avg_score_invalid = validation_results[~validation_results['Is_Valid']]['Becslés'].mean()
            st.metric("Avg Score (Invalid)", f"{avg_score_invalid:.3f}")
        with col3:
            score_range = f"{validation_results['Becslés'].min():.3f} - {validation_results['Becslés'].max():.3f}"
            st.metric("Score Range", score_range)
    
    with tab2:
        # Scatter plot for delta comparison
        st.subheader("Delta Comparison: Original vs Inverted")
        
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
        st.subheader("📊 Validation Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            with st.container(border=True):
                st.write("**Validation Pattern**")
                if validity_percentage >= 80:
                    st.success("**Strong Consistency** - Results show high reliability")
                elif validity_percentage >= 50:
                    st.warning("**Moderate Consistency** - Some results need review")
                else:
                    st.error("**Low Consistency** - Significant review required")
                
                st.write(f"**Success Rate**: {validity_percentage:.1f}%")
                st.write(f"**Valid Cases**: {valid_count}/{total_count}")
        
        with insights_col2:
            with st.container(border=True):
                st.write("**Data Quality**")
                delta_correlation = validation_results['Original_Delta'].corr(validation_results['Inverted_Delta'])
                st.write(f"**Delta Correlation**: {delta_correlation:.3f}")
                
                if abs(delta_correlation) < 0.3:
                    st.info("Low correlation expected for valid inversion")
                else:
                    st.warning("Unexpected correlation pattern detected")
    
    with tab3:
        # Enhanced results table
        st.subheader("📋 Detailed Validation Results")
        
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
            st.subheader("⚠️ Cases Requiring Review")
            
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
    st.markdown("---")
    st.subheader("💾 Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Full dataset export
        full_csv = validation_results.to_csv(index=False)
        st.download_button(
            "📊 Download Full Dataset (CSV)",
            full_csv,
            f"validation_full_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
            help="Includes all validation data and calculations"
        )
    
    with export_col2:
        # Summary report
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