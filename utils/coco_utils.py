#coco_utils.py - utility functions for COCO interaction and inverting rankings 

import requests
import pandas as pd
import streamlit as st
import re
import time
import unicodedata
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
        'object': object_names.replace("\n", "\r\n") if object_names else "",
        'attribute': attribute_names.replace("\n", "\r\n") if attribute_names else ""
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
                    df_list = pd.read_html(StringIO(table_html), header=0)
                    if df_list:
                        df = df_list[0]
                        # Check if this looks like a proper table (not just numbers as headers)
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


def _normalize_coco_label(value):
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", "", ascii_text).lower()


def _to_float_loose(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    txt = str(value).strip()
    if not txt:
        return None
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", txt)
    if not nums:
        return None
    token = nums[-1].replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def prepare_coco_matrix_df(ranked_df):
    """Return the dataframe slice that is actually submitted to COCO."""
    # Keep only the ranked attribute columns and Y_value, exclude summary columns.
    exclude_columns = ["userid", "userfullname", "Average_Rank", "Overall_Rank"]

    # Also exclude any original attribute columns (keep only _rank columns).
    original_columns = [
        col
        for col in ranked_df.columns
        if not col.endswith("_rank")
        and col not in exclude_columns
        and col != "Y_value"
    ]
    exclude_columns.extend(original_columns)
    return ranked_df.drop(columns=exclude_columns, errors="ignore")


def get_coco_rank_columns(ranked_df):
    """Return the ordered rank-column list used in the COCO input matrix."""
    matrix_df = prepare_coco_matrix_df(ranked_df)
    return [col for col in matrix_df.columns if col != "Y_value"]

def prepare_coco_matrix(ranked_df):
    """Prepare ranked data for COCO analysis"""
    import streamlit as st
    matrix_df = prepare_coco_matrix_df(ranked_df)
    
    # Debug information
    if 'show_coco_debug' in st.session_state and st.session_state.show_coco_debug:
        excluded_columns = [col for col in ranked_df.columns if col not in matrix_df.columns]
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


def get_coco_stairs_table(tables, stairs_number):
    """Locate a parsed COCO stairs table by its first-column header."""
    if not tables:
        return pd.DataFrame()

    target_keys = {
        f"lepcsok({stairs_number})",
        f"stairs({stairs_number})",
    }
    for df in tables.values():
        if df is None or df.empty or len(df.columns) == 0:
            continue
        first_col = _normalize_coco_label(df.columns[0])
        if first_col in target_keys or (
            ("lepcsok" in first_col or "stairs" in first_col) and f"({stairs_number})" in first_col
        ):
            return df
    return pd.DataFrame()


def detect_excluded_rank_columns(tables, ranked_df):
    """
    Detect exclusion candidates from the Stairs(2) table.

    The reference workflow flags rank columns whose S1 value equals n-1,
    where n is the number of ranked objects submitted to COCO.
    """
    stairs2_df = get_coco_stairs_table(tables, 2)
    if stairs2_df.empty:
        return []

    rank_columns = get_coco_rank_columns(ranked_df)
    if not rank_columns:
        return []

    threshold = float(max(len(ranked_df) - 1, 0))
    first_col = stairs2_df.columns[0]
    row_labels = stairs2_df[first_col].astype(str).map(_normalize_coco_label)
    s1_matches = stairs2_df.loc[row_labels == "s1"]
    if s1_matches.empty:
        s1_row = stairs2_df.iloc[0]
    else:
        s1_row = s1_matches.iloc[0]

    data_columns = list(stairs2_df.columns[1:])
    exclusion_candidates = []
    for idx, rank_column in enumerate(rank_columns):
        if idx >= len(data_columns):
            break
        coco_column = data_columns[idx]
        s1_value = _to_float_loose(s1_row.get(coco_column))
        if s1_value is None:
            continue
        if abs(float(s1_value) - threshold) < 1e-9:
            attribute_name = rank_column[:-5] if rank_column.endswith("_rank") else rank_column
            exclusion_candidates.append(
                {
                    "rank_column": rank_column,
                    "attribute_name": attribute_name,
                    "attribute_label": attribute_name.replace("_", " ").title(),
                    "coco_column": str(coco_column),
                    "s1_value": float(s1_value),
                    "threshold": threshold,
                    "column_index": idx + 1,
                }
            )

    return exclusion_candidates


def build_coco_rerun_frame(ranked_df, selected_rank_columns):
    """Create a ranked dataframe containing only identifiers, selected ranks, and Y."""
    selected_rank_columns = [col for col in selected_rank_columns if col in ranked_df.columns]
    identifier_columns = [col for col in ["userid", "userfullname"] if col in ranked_df.columns]
    tail_columns = ["Y_value"] if "Y_value" in ranked_df.columns else []
    ordered_columns = identifier_columns + selected_rank_columns + tail_columns

    if not selected_rank_columns:
        fallback_columns = identifier_columns + tail_columns
        return ranked_df[fallback_columns].copy() if fallback_columns else pd.DataFrame(index=ranked_df.index)
    return ranked_df[ordered_columns].copy()


def build_object_names_payload(ranked_df):
    """Create the object names payload for COCO using student names with userid fallback."""
    names = []
    for idx, row in ranked_df.iterrows():
        raw_name = row.get("userfullname")
        fallback = row.get("userid")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            candidate = fallback if not pd.isna(fallback) else f"student_{idx + 1}"
        else:
            candidate = raw_name
        cleaned = re.sub(r"[\t\r\n]+", " ", str(candidate)).strip()
        if not cleaned:
            cleaned = f"student_{idx + 1}"
        names.append(cleaned)
    return "\n".join(names)

def save_coco_debug(html_snippet):
    """Save debug information to session state"""
    debug_info = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'html_snippet': html_snippet
    }
    if 'coco_debug' not in st.session_state:
        st.session_state.coco_debug = []
    st.session_state.coco_debug.append(debug_info)
