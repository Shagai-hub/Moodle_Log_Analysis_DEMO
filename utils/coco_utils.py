import requests
import pandas as pd
import streamlit as st
import re
import time
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO

def send_coco_request(matrix_data, job_name="MyTest", stair="", object_names="", attribute_names="", keep_files=False, timeout=120):
    """
    Send a request to the COCO Y0 service with some robustness for cloud deployments.
    Returns the requests.Response object (so caller can inspect status, headers and raw bytes).
    """
    url = "https://miau.my-x.hu/myx-free/coco/engine3.php"
    session = requests.Session()
    # Use a browser-like User-Agent so the remote site doesn't return a "bot" or reduced page.
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
    # Ensure correct encoding decoding later from bytes
    # Do not change resp.encoding here — decoding will be explicit in parse function
    return resp

def parse_coco_html(resp_or_html):
    """
    Parse the HTML response from COCO and extract all tables robustly.
    Accepts either a requests.Response or a raw HTML string.
    Returns dict of { 'table_0': DataFrame, ... }.
    """
    # Prepare html string
    try:
        if hasattr(resp_or_html, "content") and hasattr(resp_or_html, "status_code"):
            # requests.Response
            raw_bytes = resp_or_html.content
            # site uses ISO-8859-2; explicitly decode with replace to avoid crashes
            html = raw_bytes.decode('iso-8859-2', errors='replace')
        else:
            html = str(resp_or_html)
    except Exception as e:
        st.error(f"Could not decode HTML response: {e}")
        return {}
    
    # Try multiple BeautifulSoup parsers depending on availability
    soup = None
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            break
        except Exception:
            continue
    
    if soup is None:
        # ultimate fallback
        soup = BeautifulSoup(html, "html.parser")
    
    # Find all <table> tags first
    tables = soup.find_all('table')
    table_dataframes = {}
    
    # If we found tables via BS, try to parse each
    if tables:
        for i, table in enumerate(tables):
            table_html = str(table)
            try:
                # For first table we may have header-less structure; use header=None as fallback
                try:
                    df_list = pd.read_html(StringIO(table_html))
                    df = df_list[0]
                except Exception as e_read:
                    # fallback: try header=None
                    df = pd.read_html(StringIO(table_html), header=None)[0]
            except Exception as e:
                # Last-resort parse: manual row/td extraction via BeautifulSoup
                try:
                    rows = []
                    for tr in table.find_all("tr"):
                        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                        if cols:
                            rows.append(cols)
                    df = pd.DataFrame(rows)
                except Exception as e_manual:
                    print(f"Could not parse table {i}: {e} / {e_manual}")
                    continue
            
            # Make columns unique & safe
            cols = list(df.columns)
            unique_cols = []
            for idx, col in enumerate(cols):
                base = str(col) if not pd.isna(col) else f"col_{idx}"
                # collapse whitespace and replace special chars
                clean = re.sub(r'[^A-Za-z0-9_]', '_', base.strip())
                # ensure uniqueness
                if clean in unique_cols:
                    suffix = 1
                    while f"{clean}_{suffix}" in unique_cols:
                        suffix += 1
                    clean = f"{clean}_{suffix}"
                unique_cols.append(clean)
            df.columns = unique_cols
            table_dataframes[f"table_{i}"] = df
    else:
        # No <table> tags found — try pd.read_html on full page (handles some malformed HTML)
        try:
            dfs = pd.read_html(StringIO(html))
            for i, df in enumerate(dfs):
                # clean column names similar to above
                cols = list(df.columns)
                unique_cols = []
                for idx, col in enumerate(cols):
                    base = str(col) if not pd.isna(col) else f"col_{idx}"
                    clean = re.sub(r'[^A-Za-z0-9_]', '_', base.strip())
                    if clean in unique_cols:
                        suffix = 1
                        while f"{clean}_{suffix}" in unique_cols:
                            suffix += 1
                        clean = f"{clean}_{suffix}"
                    unique_cols.append(clean)
                df.columns = unique_cols
                table_dataframes[f"table_{i}"] = df
        except Exception as e_full:
            # nothing found — return empty dict (caller will handle debug saving)
            print(f"pd.read_html on full page failed: {e_full}")
    
    return table_dataframes

def save_coco_debug_html(conn, html, resp=None):
    """Save debug HTML to database for troubleshooting"""
    try:
        debug_df = pd.DataFrame({
            'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")],
            'status_code': [getattr(resp, "status_code", None)],
            'url': [getattr(resp, "url", None)],
            'html_snippet': [html[:4000]]  # store first chunk to keep table size reasonable
        })
        debug_df.to_sql('coco_run_debug', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Failed to save debug HTML to DB: {e}")

def clean_column_name(name):
    """
    Clean column names for SQL compatibility
    """
    # Remove special characters and replace spaces with underscores
    if isinstance(name, str):
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return f"column_{name}"

def invert_ranking(matrix_df):
    """
    Invert the ranking: NumOfObjects - RankedValue + 1
    """
    num_objects = len(matrix_df)
    inverted_df = matrix_df.copy()
    
    # Invert all numeric columns except the last one (Y_value)
    for col in inverted_df.columns[:-1]:  # All columns except the last one
        if inverted_df[col].dtype in [np.int64, np.float64]:
            inverted_df[col] = num_objects - inverted_df[col] + 1
    
    return inverted_df

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

def save_coco_debug(html_snippet):
    """Save debug information to session state"""
    debug_info = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'html_snippet': html_snippet
    }
    if 'coco_debug' not in st.session_state:
        st.session_state.coco_debug = []
    st.session_state.coco_debug.append(debug_info)