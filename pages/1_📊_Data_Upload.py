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
    
    # Custom CSS for enhanced styling
    st.markdown("""
        <style>
        .upload-header {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .upload-header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.3rem;
            font-weight: 700;
        }
        .upload-header p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin: 0;
        }
        .upload-zone {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #4facfe;
            margin: 1rem 0;
            text-align: center;
        }
        .info-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .metric-card {
            background: Black;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Mobile Responsive */
        @media only screen and (max-width: 768px) {
            .upload-header h1 {
                font-size: 1.8rem !important;
            }
            .upload-header p {
                font-size: 0.95rem !important;
            }
            .upload-zone {
                padding: 1rem !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
        <div class="upload-header">
            <h1>📥 Upload Discussion Data</h1>
            <p>Upload your Moodle discussion data to begin analysis</p>
        </div>
    """, unsafe_allow_html=True)
    # Info section
    st.markdown("""
        <div class="info-card">
            <strong>📋 Supported Formats:</strong> CSV, XLSX<br>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Try It Instantly")
    st.write(
        "Download our clean demo datasets to explore the workflow without preparing your own files first."
    )

    demo_col_csv, demo_col_xlsx = st.columns(2)

    sample_csv_path = os.path.join("sample_data", "discussion_demo.csv")
    sample_xlsx_path = os.path.join("sample_data", "discussion_demo.xlsx")

    with demo_col_csv:
        with open(sample_csv_path, "rb") as sample_csv:
            st.download_button(
                label="📥 Download Demo CSV",
                data=sample_csv.read(),
                file_name="discussion_demo.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with demo_col_xlsx:
        with open(sample_xlsx_path, "rb") as sample_xlsx:
            st.download_button(
                label="📥 Download Demo XLSX",
                data=sample_xlsx.read(),
                file_name="discussion_demo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            

    # Upload section
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse", 
        type=["csv", "xlsx"], 
        help="File should contain columns like userid, userfullname, message, created, etc.",
        label_visibility="visible",
        width="stretch"
    )
    st.markdown("</div>", unsafe_allow_html=True)
        
    
    if uploaded_file:
        with st.spinner("🔄 Processing your file..."):
            df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            # Store in session state instead of database
            data_manager.store_raw_data(df, source_info=uploaded_file.name)
            
            st.success(f"✅ Successfully loaded '{uploaded_file.name}'!")
            
            # Show dataset overview with styled metrics
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Unique Users", df['userfullname'].nunique())
            
            # Show data preview
            st.markdown("### 📋 Data Preview")
            with st.expander("Click to view first 10 rows", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Action buttons section
            st.markdown("### 🎯 Next Steps")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Processed Data",
                    csv_data,
                    f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("⚙️ Go to Configuration", use_container_width=True, key="go_to_config", type="primary"):
                    try:
                        st.switch_page("pages/2_⚙️_Configuration.py")
                    except Exception:
                        st.warning("Unable to auto-navigate. Please open '⚙️ Configuration' from the sidebar.")


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
                except Exception:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
                if df is not None:
                    break
            except Exception:
                df = None
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