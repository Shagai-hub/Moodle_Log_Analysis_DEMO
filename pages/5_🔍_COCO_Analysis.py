import streamlit as st
import pandas as pd
from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.coco_utils import (
    send_coco_request,
    parse_coco_html,
    clean_coco_dataframe,
    prepare_coco_matrix,
    invert_ranking,
    build_object_names_payload,
)
from assets.ui_components import apply_theme, divider, info_panel, page_header, section_header, nav_footer

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()

from utils.ui_steps import render_steps
render_steps(active="1 Analyze")
apply_theme()

def main():
    data_manager = st.session_state.data_manager
    config = st.session_state.config
    
    page_header(
        "COCO Analysis",
        "Run multi-criteria evaluation on the ranked student results.",
        icon="🔍",
        align="left",
        compact=True,
    )
    
    # Check if ranking is available
    ranked_data = data_manager.get_ranked_results()
    if ranked_data is None:
        st.warning("📊 Please run student ranking first on the Ranking page.")
        divider()
        nav_footer(
            back={
                "label": "⬅️ Back to Ranking",
                "page": "pages/4_🏆_Ranking.py",
                "key": "nav_back_to_ranking_missing_coco",
                "fallback": "🏆 Ranking",
            }
        )
        return
    
    st.success(f"✅ Ready to analyze {len(ranked_data)} ranked students")

    
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Analysis Parameters", icon="📐", tight=True)
        job_name = st.text_input(
            "Analysis Name:",
            value="StudentRanking",
            help="Name for this COCO analysis run"
        )
        
        # Calculate stair value (number of objects)
        stair_value = len(ranked_data)
        st.info(f"**Stair Value (Objects):** {stair_value}")
    
    with col2:
        section_header("Data Summary", icon="📊", tight=True)
        st.metric("Students", len(ranked_data))
        st.metric("Y Value", ranked_data["Y_value"].iloc[0] if "Y_value" in ranked_data.columns else "N/A")
    
    # Data preview
    with st.expander("🔍 Preview Ranked Data", expanded=True):
        st.dataframe(ranked_data.head(10), use_container_width=True)
        st.caption(f"Full dataset: {ranked_data.shape[0]} rows × {ranked_data.shape[1]} columns")
    
    # Run COCO + Validation
    divider()
    
    coco_tables = None
    validation_results = None

    if st.button("🚀 Run COCO & Validation", type="primary", use_container_width=True):
        coco_tables = run_coco_analysis(ranked_data, job_name, stair_value)
        if coco_tables:
            data_manager.store_coco_results(coco_tables)
            st.session_state.coco_last_job_name = job_name
            coco_tables = data_manager.get_coco_results()

            if has_validation_requirements(coco_tables):
                validation_results = run_validation_analysis(ranked_data, coco_tables)
                if validation_results is not None and not validation_results.empty:
                    data_manager.store_validation_results(validation_results)
                    validation_results = data_manager.get_validation_results()
                else:
                    st.warning("Validation did not produce meaningful results. Please review the COCO output.")
            else:
                info_panel(
                    "Validation skipped because the COCO output does not include the required columns.",
                    icon="ℹ️",
                )
        else:
            st.error("COCO analysis did not return any tables. Please try again.")
    else:
        coco_tables = data_manager.get_coco_results()
        validation_results = data_manager.get_validation_results()

    if coco_tables:
        display_coco_results(coco_tables)
        col_left, col_center, col_right = st.columns([1.3, 0.9, 0.9])
        with col_center:
            if st.button(
                "Visualizations",
                key="VISUAL",
                icon="📊",
                ):
                 st.switch_page("pages/6_📊_Visualizations.py")
        if not has_validation_requirements(coco_tables):
            info_panel(
                "Validation metrics require the COCO output table containing the columns "
                "`Delta/Tény` and `Becslés`.",
                icon="⚠️",
            )
    else:
        st.info("Run the combined analysis to populate COCO results.")

    if coco_tables:
        divider()

    if validation_results is not None and not validation_results.empty:
        info_panel(
            "Detailed validation dashboards are ready. Open the Visualizations page to explore them.",
            icon="✅",
        )
        
    elif coco_tables:
        st.info("Validation results will appear automatically after a successful COCO run.")

    forward_spec = None
    if coco_tables:
        forward_spec = {
            "label": "📊 Explore Visualizations",
            "page": "pages/6_📊_Visualizations.py",
            "key": "nav_to_visualizations_from_coco",
            "fallback": "📊 Visualizations",
            "help": "Review charts without re-running heavy computations",
        }

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to Ranking",
            "page": "pages/4_🏆_Ranking.py",
            "key": "nav_back_to_ranking_footer",
            "fallback": "🏆 Ranking",
        },
        forward=forward_spec,
    )

def run_coco_analysis(ranked_data, job_name, stair_value):
    """Execute COCO analysis and return the parsed tables."""

    with st.spinner("Preparing data for COCO analysis..."):
        matrix_data = prepare_coco_matrix(ranked_data)
        object_names = build_object_names_payload(ranked_data)

    with st.expander("🔍 Matrix Data Preview", expanded=False):
        st.text_area("COCO Input Matrix (first 500 chars):", matrix_data[:500], height=150)
        st.text_area(
            "COCO Object Names (first 500 chars):",
            object_names[:500],
            height=150,
            help="Names are sent in parallel with the matrix so COCO can label objects correctly.",
        )

    st.info("🌐 Sending request to COCO service...")

    resp = None
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Connecting to COCO service...")
        progress_bar.progress(20)

        resp = send_coco_request(
            matrix_data=matrix_data,
            job_name=job_name,
            stair=str(stair_value),
            object_names=object_names,
            attribute_names="",
            keep_files=False,
            timeout=180,
        )

        status_text.text("Processing COCO response...")
        progress_bar.progress(60)

        status_text.text("Parsing analysis results...")
        progress_bar.progress(80)
        tables = parse_coco_html(resp)

        status_text.text("Finalizing results...")
        progress_bar.progress(100)

        if not tables:
            handle_coco_error(resp, "❌ COCO analysis returned no tables.")
            return None

        st.session_state.last_coco_html = decode_coco_response(resp)
        status_text.text("✅ COCO analysis completed!")
        st.success("✅ COCO analysis completed!")
        return tables

    except Exception as exc:
        handle_coco_error(resp, f"❌ COCO analysis failed: {exc}")
        st.info("Please check your internet connection and try again.")
        return None


def decode_coco_response(resp):
    """Decode raw HTML returned by the COCO service."""
    if resp is None:
        return "<no response>"
    try:
        return resp.content.decode("iso-8859-2", errors="replace")
    except Exception:
        return resp.text if hasattr(resp, "text") else "<no html>"


def handle_coco_error(resp, message=None):
    """Handle COCO service errors and expose debug information."""
    st.error(message or "❌ No analysis results received from COCO service")

    raw_html = decode_coco_response(resp)
    st.session_state.last_coco_html = raw_html

    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Response Details:**")
        st.write(f"Status Code: {getattr(resp, 'status_code', 'N/A')}")
        st.write(f"URL: {getattr(resp, 'url', 'N/A')}")

        st.write("**Response Snippet:**")
        st.code(raw_html[:2000], language="html")

        if "coco_debug" in st.session_state:
            st.write("**Parsing Debug Info:**")
            for debug in st.session_state.coco_debug[-3:]:
                st.write(f"Time: {debug['timestamp']}")
                st.code(debug["html_snippet"][:1000], language="html")

    st.info("💡 This might be a temporary service issue. Please try again in a few moments.")


def has_validation_requirements(tables):
    """Return True when the COCO output contains the columns needed for validation."""
    if not tables:
        return False
    main_table = tables.get("table_4")
    if main_table is None:
        return False
    required = {"Delta/Tény", "Becslés"}
    return required.issubset(set(main_table.columns))


def display_coco_results(tables):
    """Render COCO core results and diagnostics."""
    section_header("COCO Analysis Result", icon="📊")
    st.caption(f"Parsed {len(tables)} tables from the COCO service.")

    display_key_tables(tables)
    display_export_options(tables)

    if "last_coco_html" in st.session_state:
        with st.expander("🔎 Raw COCO HTML (from last run)", expanded=False):
            st.text_area(
                "Raw COCO HTML",
                st.session_state.last_coco_html,
                height=600,
                key="raw_coco_html",
            )
            st.download_button(
                "⬇ Download last COCO HTML",
                st.session_state.last_coco_html,
                "coco_last_response.html",
                "text/html",
                use_container_width=True,
            )


def display_key_tables(tables):
    """Highlight the key tables returned by COCO."""
    score_table = None
    ranking_table = None

    for name, df in tables.items():
        cols_lower = [str(col).lower() for col in df.columns]
        if any("becsl" in col for col in cols_lower) or any("score" in col for col in cols_lower):
            score_table = (name, df)
        if any("rank" in col for col in cols_lower):
            ranking_table = (name, df)

    if score_table:
        table_name, df = score_table
        display_df = clean_coco_dataframe(df).copy()
        score_cols = [
            col for col in display_df.columns if any(keyword in col.lower() for keyword in ["becsl", "score", "value"])
        ]
        if score_cols:
            score_col = score_cols[0]
            if pd.api.types.is_numeric_dtype(display_df[score_col]):
                display_df = display_df.sort_values(score_col, ascending=False)
        st.subheader(f"🏁 {table_name}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Score table not found in the COCO output.")

    if ranking_table and (not score_table or ranking_table[0] != score_table[0]):
        table_name, df = ranking_table
        st.subheader(f"📋 {table_name}")
        st.dataframe(clean_coco_dataframe(df), use_container_width=True)

    with st.expander("🔍 All Result Tables", expanded=False):
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(clean_coco_dataframe(df), use_container_width=True)


def display_export_options(tables):
    """Display options to export COCO results."""
    divider()
    st.header("💾 Export Results")

    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for table_name, df in tables.items():
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{table_name}.csv", csv_data)

    st.download_button(
        "📦 Download All Results (ZIP)",
        zip_buffer.getvalue(),
        "coco_results.zip",
        "application/zip",
        use_container_width=True,
    )


def run_validation_analysis(ranked_data, coco_results):
    """Run the complete validation analysis with progress feedback."""
    main_table = coco_results.get("table_4")
    if main_table is None:
        st.error("Validation requires the COCO output table named 'table_4'.")
        return None

    with st.status("🔄 **Running Validation Analysis...**", expanded=True) as status:
        try:
            matrix_df = ranked_data.drop(columns=["userid", "userfullname"], errors="ignore")
            inverted_matrix_df = invert_ranking(matrix_df)

            st.write("🔄 Converting inverted ranking to COCO input …")
            inverted_matrix_data = prepare_coco_matrix(inverted_matrix_df)

            st.write("🌐 Sending inverted matrix to COCO…")
            stair_value = len(ranked_data)
            object_names = build_object_names_payload(ranked_data)
            resp = send_coco_request(
                matrix_data=inverted_matrix_data,
                job_name="StudentRankingInverted",
                stair=str(stair_value),
                object_names=object_names,
                timeout=180,
            )

            st.write("📈 Parsing inverted COCO results…")
            inverted_tables = parse_coco_html(resp)
            if not inverted_tables or "table_4" not in inverted_tables:
                status.update(label="❌ Validation failed — inverted results missing.", state="error")
                st.error("❌ No usable tables returned from the inverted COCO analysis.")
                return None

            st.write("✅ Performing validation analysis…")
            validation_results = perform_validation(main_table, inverted_tables["table_4"], ranked_data)
            if validation_results is None or validation_results.empty:
                status.update(label="❌ Validation failed.", state="error")
                st.error("Validation failed to produce meaningful results.")
                return None

            status.update(label="✅ Validation complete!", state="complete", expanded=False)
            return validation_results

        except Exception as exc:
            status.update(label="❌ Validation error.", state="error")
            st.error(f"Validation process failed: {exc}")
            import traceback

            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
            return None


def perform_validation(original_table, inverted_table, ranked_data):
    """Validate COCO analysis results by comparing original and inverted deltas."""
    try:
        delta_col = "Delta/Tény"
        becsl_col = "Becslés"

        if delta_col not in original_table.columns or delta_col not in inverted_table.columns:
            st.error(f"❌ Delta column '{delta_col}' not found in COCO output.")
            return None
        if becsl_col not in original_table.columns:
            st.error(f"❌ Estimation column '{becsl_col}' not found in COCO output.")
            return None

        original_delta = pd.to_numeric(original_table[delta_col], errors="coerce")
        inverted_delta = pd.to_numeric(inverted_table[delta_col], errors="coerce")
        original_becsl = pd.to_numeric(original_table[becsl_col], errors="coerce")

        valid_indices = original_delta.notna() & inverted_delta.notna()
        original_delta = original_delta[valid_indices]
        inverted_delta = inverted_delta[valid_indices]
        ranked_subset = ranked_data.loc[valid_indices]
        original_becsl = original_becsl[valid_indices]

        validation_product = original_delta * inverted_delta
        is_valid = validation_product <= 0

        validation_results = ranked_subset.copy()
        validation_results["Becslés"] = original_becsl
        validation_results["Original_Delta"] = original_delta
        validation_results["Inverted_Delta"] = inverted_delta
        validation_results["Validation_Product"] = validation_product
        validation_results["Validation_Result"] = is_valid.map({True: "Valid", False: "Invalid"})
        validation_results["Is_Valid"] = is_valid
        validation_results["Final_Rank"] = (
            validation_results["Becslés"].rank(ascending=False, method="min").astype(int)
        )

        return validation_results

    except Exception as exc:
        st.error(f"❌ Validation calculation error: {exc}")
        return None




if __name__ == "__main__":
    main()
