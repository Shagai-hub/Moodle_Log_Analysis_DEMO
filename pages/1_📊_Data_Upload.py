import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from io import StringIO, BytesIO
from datetime import datetime
import base64
from urllib.parse import quote

DB_NAME = "logs.db"

def read_table(conn, table_name):
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        return pd.DataFrame()

def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


# Connect or create DB
conn = sqlite3.connect(DB_NAME, check_same_thread=False)

st.header("1.📥 Upload Discussion Data")

uploaded_file = st.file_uploader(
    "Upload your discussion data file (CSV or XLSX):", 
    type=["csv", "xlsx"], 
    help="File should contain columns like userid, userfullname, message, created, etc."
)

current_table_name = None

if uploaded_file:
    # Process the uploaded file
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
        st.stop()

    # Validate required columns
    required_columns = ['userid', 'userfullname', 'message']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("💡 Your file should contain these columns: userid, userfullname, message")
        st.stop()

    # Convert date columns if present
    if 'created' in df.columns:
        try:
            df['created'] = pd.to_datetime(df['created'], errors='coerce')
            df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.warning(f"Could not parse 'created' dates: {e}")

    if 'modified' in df.columns:
        try:
            df['modified'] = pd.to_datetime(df['modified'], errors='coerce')
            df['modified'] = df['modified'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.warning(f"Could not parse 'modified' dates: {e}")

    # Clean column names for SQLite
    df.columns = [col.replace(" ", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", "") 
                 for col in df.columns]

    # Create table name from filename
    table_name = os.path.splitext(uploaded_file.name)[0]
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
    
    # Ensure table name is valid
    if not table_name or table_name[0].isdigit():
        table_name = "discussion_data_" + table_name

    try:
        # Save to database
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        
        current_table_name = table_name
        
        st.success(f"✅ Successfully loaded '{uploaded_file.name}' as '{table_name}'")
        
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

    except Exception as e:
        st.error(f"❌ Failed to save data to database: {e}")
        st.stop()
