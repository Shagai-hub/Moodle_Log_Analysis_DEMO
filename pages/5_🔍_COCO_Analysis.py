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
    
    # Display matrix preview for debugging
    with st.expander("🔍 Matrix Data Preview", expanded=False):
        st.text_area("COCO Input Matrix (first 500 chars):", matrix_data[:500], height=150)
    
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
            keep_files=False,
            timeout=180  # Increased timeout
        )
        
        status_text.text("Processing COCO response...")
        progress_bar.progress(60)
        
        
        # Parse the response with improved parsing
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
    
    # Save full HTML for debugging
    st.session_state.last_coco_html = raw_html
    
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Response Details:**")
        st.write(f"Status Code: {getattr(resp, 'status_code', 'N/A')}")
        st.write(f"URL: {getattr(resp, 'url', 'N/A')}")
        
        st.write("**Response Snippet:**")
        st.code(raw_html[:2000], language='html')
        
        # Show debug info from session state
        if 'coco_debug' in st.session_state:
            st.write("**Parsing Debug Info:**")
            for debug in st.session_state.coco_debug[-3:]:  # Show last 3 debug entries
                st.write(f"Time: {debug['timestamp']}")
                st.code(debug['html_snippet'][:1000], language='html')
    
    st.info("💡 This might be a temporary service issue. Please try again in a few moments.")

def display_coco_results(tables, data_manager, job_name, stair_value):
    
    st.header("📊 COCO Analysis Result")
    
    # Store results in session
    data_manager.store_coco_results(tables)
    
    
    # Display key tables
    display_key_tables(tables)
    
    # Export options
    display_export_options(tables, data_manager)

    if 'last_coco_html' in st.session_state:
            with st.expander("🔎 Raw COCO HTML (from last run)", expanded=False):
                st.text_area("Raw COCO HTML", st.session_state.last_coco_html, height=600, key="raw_coco_html")
                st.download_button(
                    "⬇ Download last COCO HTML",
                    st.session_state.last_coco_html,
                    "coco_last_response.html",
                    "text/html",
                    use_container_width=True
                )

def get_table_description(table_name, df, stair_value=None):
    """Get description for COCO result tables based on content"""
    # Try to infer table purpose from column names and content
    columns_lower = [str(col).lower() for col in df.columns]
    
    if any('distance' in col for col in columns_lower) and any('ideal' in col for col in columns_lower):
        return "Distance measures and final scores"
    elif any('rank' in col for col in columns_lower):
        return "Ranking results"
    elif any('weight' in col for col in columns_lower):
        return "Weighted normalized matrix"
    elif any('normal' in col for col in columns_lower):
        return "Normalized decision matrix"
    elif stair_value is not None and len(df) == stair_value:  # guard against undefined
        return "Input data verification"
    else:
        return "Analysis results"

def display_key_tables(tables):
    """Display the most important COCO result tables"""
    
    # Try to identify the score table by content
    score_table = None
    ranking_table = None
    
    for name, df in tables.items():
        cols_lower = [str(col).lower() for col in df.columns]
        # Look for typical COCO output columns
        if any('becsl' in col for col in cols_lower) or any('score' in col for col in cols_lower):
            score_table = (name, df)
        elif any('rank' in col for col in cols_lower):
            ranking_table = (name, df)
    
    # Display scores table
    if score_table:
        name, df = score_table
        display_df = clean_coco_dataframe(df)
        
        # Try to identify score column
        score_cols = [col for col in display_df.columns if any(keyword in col.lower() for keyword in ['becsl', 'score', 'value'])]
        
        if score_cols:
            score_col = score_cols[0]
            if pd.api.types.is_numeric_dtype(display_df[score_col]):
                display_df = display_df.sort_values(score_col, ascending=False)
            
    
    
    # Display all tables in expanders for completeness
    with st.expander("🔍 All Result Tables", expanded=False):
        for name, df in tables.items():
            st.subheader(f"📄 {name}")
            st.dataframe(clean_coco_dataframe(df), use_container_width=True)

def display_export_options(tables, data_manager):
    """Display options to export COCO results"""
    
    st.markdown("---")
    st.header("💾 Export Results & Result Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export all tables as CSV - directly use download_button
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for table_name, df in tables.items():
                csv_data = df.to_csv(index=False)
                zip_file.writestr(f"{table_name}.csv", csv_data)
        
        st.download_button(
            "📦 Download All Results (ZIP)",
            zip_buffer.getvalue(),
            "coco_results.zip",
            "application/zip",
            use_container_width=True
        )
    
    with col2:
        # Proceed to validation
        if st.button("✅ Proceed to Validation", use_container_width=True):
            st.session_state.proceed_to_validation = True
            st.rerun()

if __name__ == "__main__":
    main()