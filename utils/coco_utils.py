#coco_utils.py - utility functions for COCO interaction and inverting rankings 

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
    return resp

def parse_coco_html(resp_or_html):
    """
    Parse the HTML response from COCO and extract all tables robustly.
    This version properly handles the header row issue.
    """
    # Prepare html string
    try:
        if hasattr(resp_or_html, "content") and hasattr(resp_or_html, "status_code"):
            # requests.Response
            raw_bytes = resp_or_html.content
            # Use ISO-8859-2 encoding as specified
            html = raw_bytes.decode('iso-8859-2', errors='replace')
        else:
            html = str(resp_or_html)
    except Exception as e:
        st.error(f"Could not decode HTML response: {e}")
        return {}
    
    tables = {}
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        potential_tables = soup.find_all('table')
        
        for i, table in enumerate(potential_tables):
            try:
                # Extract table HTML and parse with pandas
                table_html = str(table)
                
                # Try multiple header strategies
                df_found = None
                
                # Strategy 1: Try with header=0 (first row as header)
                try:
                    df_list = pd.read_html(StringIO(table_html), header=1)
                    if df_list:
                        df = df_list[0]
                        # Check if this looks like a proper table (not just numbers as headers)
                        first_col = str(df.columns[0]) if len(df.columns) > 0 else ""
                        if not first_col.isdigit() and 'unnamed' not in first_col.lower():
                            df_found = df
                except:
                    pass
                
                # Strategy 2: Try with header=1 (second row as header) if first failed
                if df_found is None:
                    try:
                        df_list = pd.read_html(StringIO(table_html), header=1)
                        if df_list:
                            df = df_list[0]
                            first_col = str(df.columns[0]) if len(df.columns) > 0 else ""
                            if not first_col.isdigit() and 'unnamed' not in first_col.lower():
                                df_found = df
                    except:
                        pass
                
                if df_found is not None and not df_found.empty:
                    # Clean the dataframe
                    cleaned_df = clean_coco_dataframe(df_found)
                    tables[f"table_{i}"] = cleaned_df
                    
            except Exception as e:
                # Fallback to manual parsing
                try:
                    rows = []
                    for tr in table.find_all("tr"):
                        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                        if cols:
                            rows.append(cols)
                    
                    if len(rows) > 1:
                        # Try to detect if first row is header
                        first_row = rows[0]
                        has_text_headers = any(any(c.isalpha() for c in str(cell)) for cell in first_row)
                        
                        if has_text_headers and len(rows) > 1:
                            # Use first row as header
                            df = pd.DataFrame(rows[1:], columns=first_row)
                        else:
                            df = pd.DataFrame(rows)
                        
                        tables[f"table_{i}"] = clean_coco_dataframe(df)
                except Exception as e_manual:
                    continue
        
        if tables:
            return tables
    except Exception as e:
        st.warning(f"Table parsing failed: {e}")
    
    if not tables:
        st.error("No tables could be parsed from COCO response")
        # Save full HTML for debugging
        st.session_state.last_coco_html = html
    
    return tables

def save_coco_debug_html(conn, html, resp=None):
    """Save debug HTML to database for troubleshooting"""
    try:
        debug_df = pd.DataFrame({
            'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")],
            'status_code': [getattr(resp, "status_code", None)],
            'url': [getattr(resp, "url", None)],
            'html_snippet': [html[:4000]]
        })
        debug_df.to_sql('coco_run_debug', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Failed to save debug HTML to DB: {e}")

def clean_column_name(name):
    return f"column_{name}"

def invert_ranking(matrix_df):
    """Invert the ranking: NumOfObjects - RankedValue + 1"""
    num_objects = len(matrix_df)
    inverted_df = matrix_df.copy()
    
    # Identify numeric columns (excluding user identifiers and Y_value)
    exclude_columns = ["userid", "userfullname", "Y_value", "Average_Rank", "Overall_Rank"]
    numeric_columns = []
    
    for col in inverted_df.columns:
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(inverted_df[col]):
            numeric_columns.append(col)
    
    for col in numeric_columns:
        if inverted_df[col].dtype in [np.int64, np.float64]:
            inverted_df[col] = num_objects - inverted_df[col] + 1

    return inverted_df

def clean_dataframe_columns(df):
    return df

def clean_coco_dataframe(df):
    """Clean COCO dataframe by properly handling headers and encoding"""
    if df.empty:
        return df
    df = clean_dataframe_columns(df)
    
    return df

def prepare_coco_matrix(ranked_df):
    """Prepare ranked data for COCO analysis"""
    import streamlit as st
    
    # Extract just the numeric values needed for COCO
    # Keep only the ranked attribute columns and Y_value, exclude summary columns
    
    # List all columns we want to EXCLUDE from COCO input
    exclude_columns = ["userid", "userfullname", "Average_Rank", "Overall_Rank"]
    
    # Also exclude any original attribute columns (keep only _rank columns)
    original_columns = [col for col in ranked_df.columns 
                       if not col.endswith('_rank') 
                       and col not in exclude_columns 
                       and col != "Y_value"]
    exclude_columns.extend(original_columns)
    
    # Create matrix without excluded columns
    matrix_df = ranked_df.drop(columns=exclude_columns, errors='ignore')
    
    # Debug information
    if 'show_coco_debug' in st.session_state and st.session_state.show_coco_debug:
        st.write("🔍 **COCO Matrix Preparation Debug:**")
        st.write(f"Original columns: {list(ranked_df.columns)}")
        st.write(f"Excluded columns: {exclude_columns}")
        st.write(f"Final COCO columns: {list(matrix_df.columns)}")
        st.write(f"Matrix shape: {matrix_df.shape}")
    
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