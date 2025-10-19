import streamlit as st
import pandas as pd
import os
import re
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager

def main():
    # Initialize managers from session state
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    st.header("📥 Upload Discussion Data")

    uploaded_file = st.file_uploader(
        "Upload your discussion data file (CSV or XLSX):", 
        type=["csv", "xlsx"], 
        help="File should contain columns like userid, userfullname, message, created, etc."
    )

    if uploaded_file:
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            # Store in session state instead of database
            data_manager.store_raw_data(df, source_info=uploaded_file.name)
            
            st.success(f"✅ Successfully loaded '{uploaded_file.name}'!")
            
            # Show dataset overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Unique Users", df['userfullname'].nunique())
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Export option
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇ Download Processed Data as CSV",
                csv_data,
                f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            col1, col2 = st.columns([1, 1])
            with col2:
                if st.form_submit_button("🔍 Proceed to COCO Analysis", use_container_width=True, 
                            help="Navigate to COCO Analysis page with ranked data"):
                    st.switch_page("pages/5_🔍_COCO_Analysis.py")


def process_uploaded_file(uploaded_file):
    """Process uploaded file and return cleaned DataFrame"""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    df = None

    if file_ext == ".csv":
        encodings = ["utf-8", "latin1", "cp1250", "utf-8-sig"]
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                # Try different separators
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=',')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
                break
            except Exception as e:
                continue

    elif file_ext == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            st.error("Missing 'openpyxl' library. Install: `pip install openpyxl`")
            st.stop()
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Failed to read '{uploaded_file.name}': {e}")
            st.stop()

    if df is None:
        st.error(f"❌ Failed to read '{uploaded_file.name}'. Please check the file format.")
        return None

    # Validate required columns
    required_columns = ['userid', 'userfullname', 'message']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("💡 Your file should contain these columns: userid, userfullname, message")
        return None

    # Convert date columns if present
    if 'created' in df.columns:
        try:
            df['created'] = pd.to_datetime(df['created'], errors='coerce')
            # Keep as datetime objects for analysis, don't convert to string
        except Exception as e:
            st.warning(f"Could not parse 'created' dates: {e}")

    if 'modified' in df.columns:
        try:
            df['modified'] = pd.to_datetime(df['modified'], errors='coerce')
            # Keep as datetime objects
        except Exception as e:
            st.warning(f"Could not parse 'modified' dates: {e}")

    # Clean column names (remove special characters)
    df.columns = [col.replace(" ", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", "") 
                 for col in df.columns]

    return df

if __name__ == "__main__":
    main()