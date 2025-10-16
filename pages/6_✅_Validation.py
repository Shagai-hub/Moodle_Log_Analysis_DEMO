# pages/5_✅_Validation.py
import streamlit as st
import pandas as pd
import altair as alt
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
    
    st.title("✅ Validation")
    st.markdown("Validate COCO analysis results by comparing with inverted rankings.")
    
    # Check if required data is available
    ranked_data = data_manager.get_ranked_results()
    coco_results = data_manager.get_coco_results()
    
    if ranked_data is None or coco_results is None:
        st.warning("📊 Please run COCO analysis first on the COCO Analysis page.")
        return
    
    st.success(f"✅ Ready to validate {len(ranked_data)} students")
    
    # Display data summary
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Students", len(ranked_data))
    with col2:
        st.metric("COCO Tables", len(coco_results))
    with col3:
        # Check if table_4 exists
        if 'table_4' in coco_results:
            st.metric("Main Table", "table_4 ✅")
        else:
            st.metric("Main Table", "Not Found ❌")

    # Configuration section
    st.header("⚙️ Validation Configuration")
    
    # Look for the specific table_4
    if 'table_4' in coco_results:
        main_table = coco_results['table_4']
        st.success("✅ Found main results table: **table_4**")
        
        # Show table preview with specific column check
        with st.expander("🔍 Preview table_4 Results", expanded=True):
            st.dataframe(main_table.head(10), use_container_width=True)
            st.caption(f"Table shape: {main_table.shape[0]} rows × {main_table.shape[1]} columns")
            
            # Check for required columns
            st.write("**Required Column Check:**")
            delta_found = 'Delta_T_ny' in main_table.columns
            becsl_found = 'Becsl_s' in main_table.columns
            
            if delta_found:
                st.success("✅ Found 'Delta_T_ny' column")
            else:
                st.error("❌ Missing 'Delta_T_ny' column")
                st.write("Available columns:", list(main_table.columns))
                
            if becsl_found:
                st.success("✅ Found 'Becsl_s' column")
            else:
                st.error("❌ Missing 'Becsl_s' column")
                st.write("Available columns:", list(main_table.columns))
        
        # Only enable validation if required columns exist
        if delta_found and becsl_found:
            st.success("🎯 All required columns found! Ready for validation.")
            
            # Run validation
            st.markdown("---")
            if st.button("🚀 Run Validation Analysis", type="primary", use_container_width=True):
                run_validation_analysis(ranked_data, coco_results, data_manager)
        else:
            st.error("❌ Cannot run validation - missing required columns in table_4")
            st.info("Please check your COCO analysis results and ensure table_4 contains the required columns.")
            
    else:
        st.error("❌ table_4 not found in COCO results")
        st.write("Available tables:", list(coco_results.keys()))
        st.info("Please run COCO analysis again to generate the required table_4.")

def run_validation_analysis(ranked_data, coco_results, data_manager):
    """Run the complete validation analysis"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get table_4
        main_table = coco_results['table_4']
        
        # Step 1: Prepare inverted matrix
        status_text.text("Step 1: Preparing inverted matrix...")
        progress_bar.progress(20)
        
        # Extract numeric values for inversion (exclude user identifiers)
        matrix_df = ranked_data.drop(columns=["userid", "userfullname"], errors='ignore')
        inverted_matrix_df = invert_ranking(matrix_df)
        
        # Debug info
        with st.expander("🔍 Inverted Matrix Preview", expanded=False):
            st.write("**Original Matrix:**")
            st.dataframe(matrix_df.head(), use_container_width=True)
            st.write("**Inverted Matrix:**")
            st.dataframe(inverted_matrix_df.head(), use_container_width=True)
        
        # Step 2: Convert to COCO format
        status_text.text("Step 2: Converting to COCO format...")
        progress_bar.progress(40)
        
        with st.expander("🔍 Exact Matrix Data Sent to COCO", expanded=True):
            st.text_area("Matrix Data (Tab-Separated Format)", inverted_matrix_data, height=300)
        
        inverted_matrix_data = prepare_coco_matrix(inverted_matrix_df)
        
        # Step 3: Send to COCO
        status_text.text("Step 3: Sending inverted matrix to COCO...")
        progress_bar.progress(60)
        
        stair_value = len(ranked_data)
        resp = send_coco_request(
            matrix_data=inverted_matrix_data,
            job_name="StudentRankingInverted",
            stair=str(stair_value),
            timeout=180
        )
        
        # Step 4: Parse inverted results
        status_text.text("Step 4: Parsing inverted COCO results...")
        progress_bar.progress(80)
        
        inverted_tables = parse_coco_html(resp)
        
        if not inverted_tables:
            st.error("❌ No results received from inverted COCO analysis")
            return
        
        # Step 5: Perform validation
        status_text.text("Step 5: Performing validation...")
        progress_bar.progress(90)
        
        # Look for table_4 in inverted results
        if 'table_4' in inverted_tables:
            inverted_main_table = inverted_tables['table_4']
            validation_results = perform_validation(
                main_table, inverted_main_table, ranked_data
            )
            
            # FIX: Check if validation_results is not None and not empty
            if validation_results is not None and not validation_results.empty:
                # Store validation results
                data_manager.store_validation_results(validation_results)
                
                # Display results
                display_validation_results(validation_results, data_manager)
            else:
                st.error("❌ Validation failed to produce results")
        else:
            st.error("❌ table_4 not found in inverted COCO results")
            st.write("Available inverted tables:", list(inverted_tables.keys()))
        
        progress_bar.progress(100)
        status_text.text("✅ Validation completed!")
        
    except Exception as e:
        st.error(f"❌ Validation failed: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        progress_bar.progress(0)
        
def perform_validation(original_table, inverted_table, ranked_data):
    """Simple validation - just use the columns we know should be there"""
    
    try:
        # The columns should now be properly named after clean_coco_dataframe
        delta_col = 'Delta_T_ny'  # Keep the original encoding for now
        becsl_col = 'Becsl_s'     # Keep the original encoding for now
        
        # Check if columns exist
        if delta_col not in original_table.columns or delta_col not in inverted_table.columns:
            st.error(f"❌ Delta column '{delta_col}' not found")
            st.write("Original table columns:", list(original_table.columns))
            st.write("Inverted table columns:", list(inverted_table.columns))
            return None
            
        if becsl_col not in original_table.columns:
            st.error(f"❌ Becsl column '{becsl_col}' not found") 
            st.write("Original table columns:", list(original_table.columns))
            return None
        
        st.success("✅ All required columns found!")
        
        # Convert to numeric
        original_delta = pd.to_numeric(original_table[delta_col], errors='coerce')
        inverted_delta = pd.to_numeric(inverted_table[delta_col], errors='coerce')
        original_becsl = pd.to_numeric(original_table[becsl_col], errors='coerce')
        
        # Calculate validation: original_delta * inverted_delta <= 0 is valid
        validation_product = original_delta * inverted_delta
        is_valid = validation_product <= 0
        
        # Create validation results
        validation_results = ranked_data.copy()
        validation_results['Becsl_s'] = original_becsl
        validation_results['Original_Delta'] = original_delta
        validation_results['Inverted_Delta'] = inverted_delta
        validation_results['Validation_Product'] = validation_product
        validation_results['Validation_Result'] = is_valid.map({True: 'Valid', False: 'Invalid'})
        validation_results['Is_Valid'] = is_valid
        
        # Add ranking based on BecslÃ©s score
        validation_results['Final_Rank'] = validation_results['Becsl_s'].rank(ascending=False, method='min').astype(int)
        
        return validation_results
        
    except Exception as e:
        st.error(f"❌ Error in perform_validation: {str(e)}")
        return None

def display_validation_results(validation_results, data_manager):
    """Display validation results and charts"""
    
    st.header("📊 Validation Results")
    
    # Validation summary
    valid_count = validation_results['Is_Valid'].sum()
    total_count = len(validation_results)
    validity_percentage = (valid_count / total_count) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", total_count)
    with col2:
        st.metric("Valid Results", f"{valid_count}/{total_count}")
    with col3:
        st.metric("Validity Rate", f"{validity_percentage:.1f}%")
    with col4:
        status = "✅ PASS" if validity_percentage >= 80 else "⚠️ REVIEW" if validity_percentage >= 50 else "❌ FAIL"
        st.metric("Validation Status", status)
    
    # Show top performers
    st.subheader("🏆 Top 10 Performers")
    top_10 = validation_results.nlargest(10, 'Becsl_s')[['userfullname', 'Becsl_s', 'Final_Rank', 'Validation_Result']]
    top_10_display = top_10.rename(columns={
        "userfullname": "Student Name",
        "Becsl_s": "Final Score", 
        "Final_Rank": "Rank",
        "Validation_Result": "Validation"
    })
    st.dataframe(top_10_display, use_container_width=True)
    
    # Display full results table
    st.subheader("📋 Detailed Validation Results")
    
    display_columns = ["userfullname", "Final_Rank", "Becsl_s", "Validation_Result", "Original_Delta", "Inverted_Delta"]
    display_df = validation_results[display_columns].copy()
    display_df = display_df.rename(columns={
        "userfullname": "Student Name",
        "Final_Rank": "Rank",
        "Becsl_s": "Final Score",
        "Validation_Result": "Validation",
        "Original_Delta": "Original Delta",
        "Inverted_Delta": "Inverted Delta"
    })
    
    # Sort by rank
    display_df = display_df.sort_values("Rank")
    st.dataframe(display_df, use_container_width=True)
    
    # Visualization
    st.subheader("📈 Validation Visualization")
    
    # Create bar chart
    chart_df = validation_results[["userfullname", "Becsl_s", "Validation_Result", "Final_Rank"]].copy()
    chart_df = chart_df.rename(columns={
        "userfullname": "Student",
        "Becsl_s": "Score",
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
    
    # Show invalid cases for review
    invalid_cases = validation_results[~validation_results['Is_Valid']]
    if not invalid_cases.empty:
        st.subheader("⚠️ Cases Needing Review (Invalid Validation)")
        invalid_display = invalid_cases[["userfullname", "Final_Rank", "Becsl_s", "Original_Delta", "Inverted_Delta", "Validation_Product"]]
        invalid_display = invalid_display.rename(columns={
            "userfullname": "Student Name",
            "Final_Rank": "Rank",
            "Becsl_s": "Final Score",
            "Original_Delta": "Original Delta", 
            "Inverted_Delta": "Inverted Delta",
            "Validation_Product": "Product"
        })
        st.dataframe(invalid_display, use_container_width=True)
        
        # Analysis of invalid cases
        st.write("**Analysis of Invalid Cases:**")
        st.write(f"- Number of invalid cases: {len(invalid_cases)}")
        st.write(f"- Average score of invalid cases: {invalid_cases['Becsl_s'].mean():.3f}")
        st.write(f"- Range of scores in invalid cases: {invalid_cases['Becsl_s'].min():.3f} to {invalid_cases['Becsl_s'].max():.3f}")
    
    # Export options
    st.markdown("---")
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download validation results
        csv_data = validation_results.to_csv(index=False)
        st.download_button(
            "📥 Download Validation Results (CSV)",
            csv_data,
            f"validation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Run Another Analysis", use_container_width=True):
            # Clear relevant session state to start over
            keys_to_clear = ['proceed_to_validation', 'selected_attributes']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()