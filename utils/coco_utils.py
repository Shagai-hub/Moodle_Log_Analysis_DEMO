import requests
import pandas as pd
import streamlit as st
import re
import time
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO

def send_coco_request(matrix_data, job_name="StudentAnalysis", stair="", object_names="", attribute_names="", keep_files=False, timeout=120):
    """
    Send a request to the COCO Y0 service.
    Returns the requests.Response object.
    """
    url = "https://miau.my-x.hu/myx-free/coco/engine3.php"
    session = requests.Session()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://miau.my-x.hu/",
    }

    data = {
        'job': job_name,
        'matrix': matrix_data.replace("\n", "\r\n"),
        'stair': stair,
        'modell': 'Y0',
        'object': object_names,
        'attribute': attribute_names
    }
    
    if keep_files:
        data['fajl_megtart'] = '1'

    resp = session.post(url, data=data, headers=headers, timeout=timeout, allow_redirects=True)
    return resp

def parse_coco_html(resp_or_html):
    """
    Parse the HTML response from COCO and extract ALL tables.
    Returns dict of { 'table_0': DataFrame, ... }.
    """
    try:
        if hasattr(resp_or_html, "content") and hasattr(resp_or_html, "status_code"):
            raw_bytes = resp_or_html.content
            html = raw_bytes.decode('iso-8859-2', errors='replace')
        else:
            html = str(resp_or_html)
    except Exception as e:
        st.error(f"Could not decode HTML response: {e}")
        return {}

    # Try multiple parsers
    soup = None
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            break
        except Exception:
            continue
    
    if soup is None:
        soup = BeautifulSoup(html, "html.parser")
    
    table_dataframes = {}
    
    # METHOD 1: Try pd.read_html on entire HTML first (most reliable)
    try:
        dfs = pd.read_html(StringIO(html))
        for i, df in enumerate(dfs):
            if not df.empty:
                df = clean_dataframe_columns(df)
                table_dataframes[f"table_{i}"] = df
        if table_dataframes:
            st.info(f"✅ Found {len(table_dataframes)} tables using direct HTML parsing")
            return table_dataframes
    except Exception as e:
        st.warning(f"Direct HTML parsing failed: {e}")
    
    # METHOD 2: Fallback to BeautifulSoup table extraction
    tables = soup.find_all('table')
    if tables:
        st.info(f"🔍 Found {len(tables)} table tags, parsing individually...")
        
        for i, table in enumerate(tables):
            try:
                # Try multiple parsing methods for each table
                df = parse_single_table(table, i)
                if df is not None and not df.empty:
                    df = clean_dataframe_columns(df)
                    table_dataframes[f"table_{i}"] = df
                    st.success(f"✅ Successfully parsed table_{i} ({df.shape[0]}x{df.shape[1]})")
            except Exception as e:
                st.warning(f"❌ Failed to parse table_{i}: {e}")
                continue
    
    if not table_dataframes:
        st.error("❌ No tables could be parsed from COCO response")
        # Save debug info
        save_coco_debug(html[:5000])  # First 5000 chars for debugging
    
    return table_dataframes

def parse_single_table(table, table_index):
    """Parse a single table element using multiple methods"""
    table_html = str(table)
    
    # Method 1: Try pd.read_html with header detection
    try:
        dfs = pd.read_html(StringIO(table_html))
        if dfs:
            return dfs[0]
    except Exception as e1:
        pass
    
    # Method 2: Try pd.read_html without headers
    try:
        dfs = pd.read_html(StringIO(table_html), header=None)
        if dfs:
            return dfs[0]
    except Exception as e2:
        pass
    
    # Method 3: Manual parsing as last resort
    try:
        rows = []
        for tr in table.find_all("tr"):
            cols = []
            for td in tr.find_all(["td", "th"]):
                # Get text and clean it
                text = td.get_text(strip=True)
                # Remove extra whitespace and newlines
                text = re.sub(r'\s+', ' ', text)
                cols.append(text)
            if cols:  # Only add non-empty rows
                rows.append(cols)
        
        if rows:
            # Use first row as header if it looks like headers
            if any(any(c.isalpha() for c in cell) for cell in rows[0]):
                df = pd.DataFrame(rows[1:], columns=rows[0])
            else:
                df = pd.DataFrame(rows)
            return df
    except Exception as e3:
        st.warning(f"Manual parsing failed for table_{table_index}: {e3}")
    
    return None

def clean_dataframe_columns(df):
    """Clean and standardize dataframe column names"""
    if df.empty:
        return df
    
    # Clean each column name
    clean_columns = []
    for idx, col in enumerate(df.columns):
        if isinstance(col, str):
            # Remove special characters and normalize
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col.strip())
            # Remove multiple underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            # Ensure it's not empty
            if not clean_col:
                clean_col = f"column_{idx}"
        else:
            clean_col = f"column_{idx}"
        
        # Ensure uniqueness
        if clean_col in clean_columns:
            suffix = 1
            while f"{clean_col}_{suffix}" in clean_columns:
                suffix += 1
            clean_col = f"{clean_col}_{suffix}"
        
        clean_columns.append(clean_col)
    
    df.columns = clean_columns
    return df

def clean_coco_dataframe(df):
    """Additional cleaning for COCO dataframes"""
    if df.empty:
        return df
    
    df = clean_dataframe_columns(df)
    
    # Try to identify and clean numeric columns
    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Try to convert to numeric, coerce errors to NaN
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    return df

def prepare_coco_matrix(ranked_df):
    """Prepare ranked data for COCO analysis"""
    # Extract just the numeric values (without userid and userfullname)
    matrix_df = ranked_df.drop(columns=["userid", "userfullname"], errors='ignore')
    
    # Convert to string with tab separation
    matrix_lines = []
    for _, row in matrix_df.iterrows():
        row_str = "\t".join(str(val) for val in row)
        matrix_lines.append(row_str)
    
    matrix_data = "\n".join(matrix_lines)
    return matrix_data

def invert_ranking(matrix_df):
    """
    Invert the ranking: NumOfObjects - RankedValue + 1
    """
    num_objects = len(matrix_df)
    inverted_df = matrix_df.copy()
    
    # Invert all numeric columns except the last one (Y_value)
    for col in inverted_df.columns[:-1]:
        if pd.api.types.is_numeric_dtype(inverted_df[col]):
            inverted_df[col] = num_objects - inverted_df[col] + 1
    
    return inverted_df

def save_coco_debug(html_snippet):
    """Save debug information to session state"""
    debug_info = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'html_snippet': html_snippet
    }
    if 'coco_debug' not in st.session_state:
        st.session_state.coco_debug = []
    st.session_state.coco_debug.append(debug_info)