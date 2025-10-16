import streamlit as st
import pandas as pd
import time
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.coco_utils import send_coco_request, parse_coco_html, clean_coco_dataframe, prepare_coco_matrix

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    st.title("🔍 COCO Analysis")
    st.markdown("Run COCO multi-criteria analysis on ranked student data.")
    
    # Check if ranking is available
    ranked_data = data_manager.get_ranked_results()
    if ranked_data is None:
        st.warning("📊 Please run student ranking first on the Ranking page.")
        return
    
    st.success(f"✅ Ready to analyze {len(ranked_data)} ranked students")
    
    # Configuration section
    st.header("⚙️ COCO Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Analysis Parameters")
        job_name = st.text_input(
            "Analysis Name:",
            value="StudentRanking",
            help="Name for this COCO analysis run"
        )
        
        # Calculate stair value (number of objects)
        stair_value = len(ranked_data)
        st.info(f"**Stair Value (Objects):** {stair_value}")
    
    with col2:
        st.subheader("📊 Data Summary")
        st.metric("Students", len(ranked_data))
        st.metric("Attributes", len([col for col in ranked_data.columns if col.endswith('_rank')]))
        st.metric("Y Value", ranked_data["Y_value"].iloc[0] if "Y_value" in ranked_data.columns else "N/A")
    
    # Data preview
    with st.expander("🔍 Preview Ranked Data", expanded=False):
        st.dataframe(ranked_data.head(10), use_container_width=True)
        st.caption(f"Full dataset: {ranked_data.shape[0]} rows × {ranked_data.shape[1]} columns")
    
    # Run COCO analysis
    st.markdown("---")
    
    if st.button("🚀 Run COCO Analysis", type="primary", use_container_width=True):
        run_coco_analysis(ranked_data, job_name, stair_value, data_manager)

def run_coco_analysis(ranked_data, job_name, stair_value, data_manager):
    """Execute COCO analysis and display results"""
    
    # Prepare data for COCO
    with st.spinner("Preparing data for COCO analysis..."):
        matrix_data = prepare_coco_matrix(ranked_data)
    
    # Send request to COCO service
    st.info("🌐 Sending request to COCO service...")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Connecting to COCO service...")
        progress_bar.progress(20)
        
        resp = send_coco_request(
            matrix_data=matrix_data,
            job_name=job_name,
            stair=str(stair_value),
            object_names="",
            attribute_names="",
            keep_files=False
        )
        
        status_text.text("Processing COCO response...")
        progress_bar.progress(60)
        
        st.success(f"✅ COCO service responded (Status: {resp.status_code})")
        
        # Parse the response
        status_text.text("Parsing analysis results...")
        progress_bar.progress(80)
        
        tables = parse_coco_html(resp)
        
        status_text.text("Finalizing results...")
        progress_bar.progress(100)
        
        if not tables:
            handle_coco_error(resp)
            return
        
        # Process and display results
        display_coco_results(tables, data_manager, job_name, stair_value)
        
        status_text.text("✅ COCO analysis completed!")
        
    except Exception as e:
        st.error(f"❌ COCO analysis failed: {str(e)}")
        st.info("Please check your internet connection and try again.")

def handle_coco_error(resp):
    """Handle COCO service errors"""
    st.error("❌ No analysis results received from COCO service")
    
    try:
        raw_html = resp.content.decode('iso-8859-2', errors='replace')
    except Exception:
        raw_html = resp.text if hasattr(resp, 'text') else "<no html>"
    
    # Save debug information to session
    debug_info = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'status_code': getattr(resp, "status_code", None),
        'html_snippet': raw_html[:2000]
    }
    
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Response Details:**")
        st.json(debug_info)
        
        st.write("**Response Snippet:**")
        st.code(raw_html[:1000], language='html')
    
    st.info("💡 This might be a temporary service issue. Please try again in a few moments.")

def display_coco_results(tables, data_manager, job_name, stair_value):
    """Display COCO analysis results"""
    
    st.header("📊 COCO Analysis Results")
    st.success(f"✅ Analysis completed! Found {len(tables)} result tables")
    
    # Store results in session
    data_manager.store_coco_results(tables)
    
    # Display table summary
    st.subheader("📋 Result Tables")
    
    summary_data = []
    for name, df in tables.items():
        summary_data.append({
            "Table": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Description": get_table_description(name)
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Display key tables
    display_key_tables(tables)
    
    # Export options
    display_export_options(tables, data_manager)

def get_table_description(table_name):
    """Get description for COCO result tables"""
    descriptions = {
        "table_0": "Input data verification",
        "table_1": "Normalized decision matrix", 
        "table_2": "Weighted normalized matrix",
        "table_3": "Ideal and negative-ideal solutions",
        "table_4": "Distance measures and final scores",
        "table_5": "Ranking results"
    }
    return descriptions.get(table_name, "Analysis results")

def display_key_tables(tables):
    """Display the most important COCO result tables"""
    
    # Table 4 - Usually contains the final scores (Becsl_s)
    if 'table_4' in tables:
        st.subheader("🎯 Final Scores (Becsl_s)")
        table_4 = clean_coco_dataframe(tables['table_4'])
        
        # Try to find the score column
        score_columns = [col for col in table_4.columns if 'becsl' in col.lower() or 'score' in col.lower() or 'value' in col.lower()]
        
        if score_columns:
            display_df = table_4.copy()
            if 'object' in display_df.columns or 'obj' in display_df.columns:
                # Sort by score (assuming higher is better)
                score_col = score_columns[0]
                if pd.api.types.is_numeric_dtype(display_df[score_col]):
                    display_df = display_df.sort_values(score_col, ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show top performers
            if len(display_df) > 0 and score_columns:
                top_5 = display_df.head(5)
                st.subheader("🏆 Top Performers (COCO Scores)")
                for idx, row in top_5.iterrows():
                    score_val = row[score_columns[0]]
                    obj_name = row.get('object', row.get('obj', f"Object {idx+1}"))
                    st.write(f"**{idx+1}.** {obj_name} - Score: {score_val:.4f}")
        else:
            st.dataframe(table_4, use_container_width=True)
    
    # Table 5 - Usually contains rankings
    if 'table_5' in tables:
        st.subheader("📈 Final Rankings")
        table_5 = clean_coco_dataframe(tables['table_5'])
        st.dataframe(table_5, use_container_width=True)

def display_export_options(tables, data_manager):
    """Display options to export COCO results"""
    
    st.markdown("---")
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export all tables as ZIP
        if st.button("📦 Download All Results (ZIP)", use_container_width=True):
            # This would require creating a ZIP file - for now we'll do individual CSVs
            st.info("Individual CSV downloads available below")
    
    with col2:
        # Proceed to validation
        if st.button("✅ Proceed to Validation", use_container_width=True):
            st.session_state.proceed_to_validation = True
            st.rerun()
    
    # Individual table downloads
    st.subheader("📥 Download Individual Tables")
    
    for table_name, df in tables.items():
        clean_df = clean_coco_dataframe(df)
        csv_data = clean_df.to_csv(index=False)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{table_name}** - {df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.download_button(
                f"Download {table_name}",
                csv_data,
                f"{table_name}_results.csv",
                "text/csv",
                key=f"download_{table_name}",
                use_container_width=True
            )

if __name__ == "__main__":
    main()