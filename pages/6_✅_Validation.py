# pages/5_✅_Validation.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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
    
    # Add debug mode toggle
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False, 
                           help="Show detailed step-by-step computation logs")

    # Look for the specific table_4
    if 'table_4' in coco_results:
        main_table = coco_results['table_4']
        
        # Show table preview with specific column check
        with st.expander("🔍 Preview table_4 Results", expanded=True):
            st.dataframe(main_table.head(10), use_container_width=True)
            st.caption(f"Table shape: {main_table.shape[0]} rows × {main_table.shape[1]} columns")
            
            # Check for required columns
            st.write("**Required Column Check:**")
            delta_found = 'Delta/Tény' in main_table.columns
            becsl_found = 'Becslés' in main_table.columns
            
            if delta_found:
                st.success("✅ Found Delta column")
            else:
                st.error("❌ Missing 'Delta_T_ny' column")
                st.write("Available columns:", list(main_table.columns))
                
            if becsl_found:
                st.success("✅ Found estimation column")
            else:
                st.error("❌ Missing 'Becsl_s' column")
                st.write("Available columns:", list(main_table.columns))
        
        # Only enable validation if required columns exist
        if delta_found and becsl_found:
            st.success("🎯 All required columns found! Ready for validation.")
            
            # Run validation
            st.markdown("---")
            if st.button("🚀 Run Validation Analysis", type="primary", use_container_width=True):
                run_validation_analysis(ranked_data, coco_results, data_manager, debug_mode)
        else:
            st.error("❌ Cannot run validation - missing required columns in table_4")
            st.info("Please check your COCO analysis results and ensure table_4 contains the required columns.")
            
    else:
        st.error("❌ table_4 not found in COCO results")
        st.write("Available tables:", list(coco_results.keys()))
        st.info("Please run COCO analysis again to generate the required table_4.")
        
        
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col2:
            if st.button("🤖 AI Insights", use_container_width=True, help="AI insights"):
              st.switch_page("pages/7_🤖_AI_Insights.py")

def run_validation_analysis(ranked_data, coco_results, data_manager, debug_mode=False):
    """Run the complete validation analysis with detailed debugging"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create debug expander
    debug_expander = st.expander("🔍 Debug Logs", expanded=debug_mode)
    
    try:
        # Step 1: Get table_4 and prepare inverted matrix
        status_text.text("Step 1: Preparing inverted matrix...")
        progress_bar.progress(20)
        
        if debug_mode:
            with debug_expander:
                st.subheader("🔧 Step 1: Matrix Preparation")
                st.write("**Original ranked data structure:**")
                st.write(f"- Shape: {ranked_data.shape}")
                st.write(f"- Columns: {list(ranked_data.columns)}")
                st.write("**First 5 rows of original data:**")
                st.dataframe(ranked_data.head())
        
        # Get table_4
        main_table = coco_results['table_4']
        
        # Extract numeric values for inversion (exclude user identifiers)
        matrix_df = ranked_data.drop(columns=["userid", "userfullname"], errors='ignore')
        
        if debug_mode:
            with debug_expander:
                st.write("**Matrix after dropping identifiers:**")
                st.write(f"- Shape: {matrix_df.shape}")
                st.write(f"- Columns: {list(matrix_df.columns)}")
                st.write("**Matrix values before inversion:**")
                st.dataframe(matrix_df.head())
        
        # Perform matrix inversion
        inverted_matrix_df = invert_ranking(matrix_df)
        
        if debug_mode:
            with debug_expander:
                st.write("**Matrix after inversion:**")
                st.write("**Inversion Formula:** `inverted_value = max_value + min_value - original_value`")
                st.write("**Computation details:**")
                st.write(f"- Max value in matrix: {matrix_df.values.max()}")
                st.write(f"- Min value in matrix: {matrix_df.values.min()}")
                st.write("**Sample inversion calculations:**")
                sample_original = matrix_df.iloc[0, 0]
                sample_inverted = inverted_matrix_df.iloc[0, 0]
                st.write(f"Sample: {sample_original} → {sample_inverted}")
                st.dataframe(inverted_matrix_df.head())
    
        # Step 2: Convert to COCO format
        status_text.text("Step 2: Converting to COCO format...")
        progress_bar.progress(40)
        
        inverted_matrix_data = prepare_coco_matrix(inverted_matrix_df)
        
        if debug_mode:
            with debug_expander:
                st.subheader("🔧 Step 2: COCO Format Preparation")
                st.write("**COCO matrix data structure:**")
                st.write(f"- Type: {type(inverted_matrix_data)}")
                st.write(f"- Length: {len(inverted_matrix_data)}")
                st.write("**First few rows of COCO data:**")
                st.text("\n".join(inverted_matrix_data[:5]))

        # Step 3: Send to COCO
        status_text.text("Step 3: Sending inverted matrix to COCO...")
        progress_bar.progress(60)
        
        stair_value = len(ranked_data)
        if debug_mode:
            with debug_expander:
                st.subheader("🔧 Step 3: COCO API Request")
                st.write("**Request parameters:**")
                st.write(f"- Job name: StudentRankingInverted")
                st.write(f"- Stair value: {stair_value}")
                st.write(f"- Matrix rows: {len(inverted_matrix_data)}")
                st.write(f"- Timeout: 180 seconds")
        
        resp = send_coco_request(
            matrix_data=inverted_matrix_data,
            job_name="StudentRankingInverted",
            stair=str(stair_value),
            timeout=180
        )
        
        if debug_mode:
            with debug_expander:
                st.write("**COCO Response received:**")
                st.write(f"- Response type: {type(resp)}")
                if resp is not None:
                    # FIX: Properly handle Response object
                    st.write(f"- Response status code: {resp.status_code}")
                    st.write(f"- Response headers: {dict(resp.headers)}")
                    # Only show content length if available
                    if hasattr(resp, 'content'):
                        st.write(f"- Response content length: {len(resp.content) if resp.content else 0}")
                    if hasattr(resp, 'text'):
                        st.write(f"- Response text length: {len(resp.text) if resp.text else 0}")
                    # Show a preview of the response text (first 500 chars)
                    if hasattr(resp, 'text') and resp.text:
                        st.write("**Response text preview (first 500 chars):**")
                        st.text(resp.text[:500] + "..." if len(resp.text) > 500 else resp.text)
                else:
                    st.write("- Response: None")
        
        # Step 4: Parse inverted results
        status_text.text("Step 4: Parsing inverted COCO results...")
        progress_bar.progress(80)
        
        # FIX: Check if response is valid before parsing
        if resp is None:
            st.error("❌ No response received from COCO API")
            return
            
        if resp.status_code != 200:
            st.error(f"❌ COCO API returned error status: {resp.status_code}")
            if debug_mode:
                with debug_expander:
                    st.write("**Error response details:**")
                    st.text(resp.text)
            return
        
        inverted_tables = parse_coco_html(resp)
        
        if debug_mode:
            with debug_expander:
                st.subheader("🔧 Step 4: Parse COCO Results")
                st.write("**Parsed tables:**")
                st.write(f"- Number of tables found: {len(inverted_tables) if inverted_tables else 0}")
                if inverted_tables:
                    st.write(f"- Table keys: {list(inverted_tables.keys())}")
        
        if not inverted_tables:
            st.error("❌ No results received from inverted COCO analysis")
            return
        
        # Step 5: Perform validation
        status_text.text("Step 5: Performing validation...")
        progress_bar.progress(90)
        
        # Look for table_4 in inverted results
        if 'table_4' in inverted_tables:
            inverted_main_table = inverted_tables['table_4']
            
            if debug_mode:
                with debug_expander:
                    st.subheader("🔧 Step 5: Validation Computation")
                    st.write("**Tables for validation:**")
                    st.write("- Original table_4 shape:", main_table.shape)
                    st.write("- Inverted table_4 shape:", inverted_main_table.shape)
                    st.write("**Original table_4 columns:**", list(main_table.columns))
                    st.write("**Inverted table_4 columns:**", list(inverted_main_table.columns))
            
            validation_results = perform_validation(
                main_table, inverted_main_table, ranked_data, debug_mode, debug_expander
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
            if inverted_tables:
                st.write("Available inverted tables:", list(inverted_tables.keys()))
            else:
                st.write("No inverted tables available")
        
        progress_bar.progress(100)
        status_text.text("✅ Validation completed!")
        
    except Exception as e:
        st.error(f"❌ Validation failed: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        progress_bar.progress(0)
        
def perform_validation(original_table, inverted_table, ranked_data, debug_mode=False, debug_expander=None):
    """Validate COCO analysis results by comparing original and inverted deltas with detailed debugging."""
    try:
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("## 🧮 Validation Mathematical Process")
                st.write("**Validation Logic:**")
                st.write("We validate by checking if: `original_delta × inverted_delta ≤ 0`")
                st.write("This means the deltas should have opposite signs (one positive, one negative)")
                st.write("**Reasoning:** If the ranking is consistent, inverting the matrix should produce opposite delta values.")
        
        # The columns should now be properly named after clean_coco_dataframe
        delta_col = 'Delta/Tény'  # Keep the original encoding for now
        becsl_col = 'Becslés'     # Keep the original encoding for now
        
        # Check if columns exist
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("### 🔍 Step 1: Column Verification")
                st.write(f"Looking for delta column: '{delta_col}'")
                st.write(f"Looking for becsl column: '{becsl_col}'")
        
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
        
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("✅ Columns found successfully")
                st.write("### 🔢 Step 2: Data Type Conversion")
                st.write("Converting delta and becsl columns to numeric types...")
        
        # Convert to numeric and handle invalid values
        original_delta = pd.to_numeric(original_table[delta_col], errors='coerce')
        inverted_delta = pd.to_numeric(inverted_table[delta_col], errors='coerce')
        original_becsl = pd.to_numeric(original_table[becsl_col], errors='coerce')
        
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("**Data Conversion Results:**")
                st.write(f"- Original delta: {len(original_delta)} values, {original_delta.isna().sum()} NaN")
                st.write(f"- Inverted delta: {len(inverted_delta)} values, {inverted_delta.isna().sum()} NaN")
                st.write(f"- Original becsl: {len(original_becsl)} values, {original_becsl.isna().sum()} NaN")
                st.write("**Sample converted values:**")
                st.write(f"Original delta sample: {original_delta.head().tolist()}")
                st.write(f"Inverted delta sample: {inverted_delta.head().tolist()}")
                st.write(f"Original becsl sample: {original_becsl.head().tolist()}")
        
        # Ensure no NaN values in deltas
        if original_delta.isna().any() or inverted_delta.isna().any():
            st.warning("⚠️ Some delta values could not be converted to numeric. These rows will be excluded from validation.")
        
        # Filter out rows with NaN values
        valid_indices = original_delta.notna() & inverted_delta.notna()
        original_delta = original_delta[valid_indices]
        inverted_delta = inverted_delta[valid_indices]
        ranked_data = ranked_data.loc[valid_indices]
        original_becsl = original_becsl[valid_indices]
        
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("### 🧹 Step 3: Data Cleaning")
                st.write(f"Removed {(~valid_indices).sum()} rows with NaN values")
                st.write(f"Remaining valid rows: {valid_indices.sum()}")
                st.write("**After cleaning:**")
                st.write(f"- Original delta: {len(original_delta)} values")
                st.write(f"- Inverted delta: {len(inverted_delta)} values")
                st.write(f"- Ranked data: {len(ranked_data)} rows")
                st.write(f"- Original becsl: {len(original_becsl)} values")
        
        # Calculate validation: original_delta * inverted_delta <= 0 is valid
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("### 🧮 Step 4: Validation Computation")
                st.write("**Mathematical Operation:** `validation_product = original_delta × inverted_delta`")
                st.write("**Validation Condition:** `validation_product ≤ 0`")
                st.write("This means the deltas should have opposite signs for valid results")
        
        validation_product = original_delta * inverted_delta
        is_valid = validation_product <= 0
        
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("**Validation Computation Results:**")
                st.write(f"- Validation product range: [{validation_product.min():.3f}, {validation_product.max():.3f}]")
                st.write(f"- Valid cases (product ≤ 0): {is_valid.sum()}")
                st.write(f"- Invalid cases (product > 0): {(~is_valid).sum()}")
                st.write("**Sample computations:**")
                sample_df = pd.DataFrame({
                    'Original Delta': original_delta.head(),
                    'Inverted Delta': inverted_delta.head(),
                    'Product': validation_product.head(),
                    'Valid': is_valid.head()
                })
                st.dataframe(sample_df)
        
        # Create validation results
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("### 📊 Step 5: Results Assembly")
                st.write("Creating final validation results dataframe...")
        
        validation_results = ranked_data.copy()
        validation_results['Becslés'] = original_becsl
        validation_results['Original_Delta'] = original_delta
        validation_results['Inverted_Delta'] = inverted_delta
        validation_results['Validation_Product'] = validation_product
        validation_results['Validation_Result'] = is_valid.map({True: 'Valid', False: 'Invalid'})
        validation_results['Is_Valid'] = is_valid
        
        # Add ranking based on Becsl_s score
        validation_results['Final_Rank'] = validation_results['Becslés'].rank(ascending=False, method='min').astype(int)
        
        if debug_mode and debug_expander:
            with debug_expander:
                st.write("### 🎯 Step 6: Final Ranking")
                st.write("**Ranking Method:** `rank(ascending=False, method='min')`")
                st.write("Higher Becslés values get better (lower) ranks")
                st.write("**Final results structure:**")
                st.write(f"- Total rows: {len(validation_results)}")
                st.write(f"- Columns: {list(validation_results.columns)}")
                st.write("**Sample final results:**")
                st.dataframe(validation_results[['userfullname', 'Becslés', 'Final_Rank', 'Validation_Result']].head())
        
        return validation_results
        
    except Exception as e:
        st.error(f"❌ Error in perform_validation: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
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
    
    
    # Display full results table
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
    
    # Sort by rank
    display_df = display_df.sort_values("Rank")
    st.dataframe(display_df, use_container_width=True)
    
    # Visualization
    st.subheader("📈 Validation Visualization")
    
    # Create bar chart
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
    
    # Show invalid cases for review
    invalid_cases = validation_results[~validation_results['Is_Valid']]
    if not invalid_cases.empty:
        st.subheader("⚠️ Cases Needing Review (Invalid Validation)")
        invalid_display = invalid_cases[["userfullname", "Final_Rank", "Becslés", "Original_Delta", "Inverted_Delta", "Validation_Product"]]
        invalid_display = invalid_display.rename(columns={
            "userfullname": "Student Name",
            "Final_Rank": "Rank",
            "Becslés": "Final Score",
            "Original_Delta": "Original Delta", 
            "Inverted_Delta": "Inverted Delta",
            "Validation_Product": "Product"
        })
        st.dataframe(invalid_display, use_container_width=True)
        
        # Analysis of invalid cases
        st.write("**Analysis of Invalid Cases:**")
        st.write(f"- Number of invalid cases: {len(invalid_cases)}")
        st.write(f"- Average score of invalid cases: {invalid_cases['Becslés'].mean():.3f}")
        st.write(f"- Range of scores in invalid cases: {invalid_cases['Becslés'].min():.3f} to {invalid_cases['Becslés'].max():.3f}")
        
        # Mathematical analysis of invalid cases
        st.write("**Mathematical Analysis of Invalid Cases:**")
        st.write(f"- Average validation product: {invalid_cases['Validation_Product'].mean():.3f}")
        st.write(f"- Product range: [{invalid_cases['Validation_Product'].min():.3f}, {invalid_cases['Validation_Product'].max():.3f}]")
        st.write("**Interpretation:** Positive products indicate both deltas have the same sign (both positive or both negative)")
    
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


if __name__ == "__main__":
    main()