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
    get_coco_rank_columns,
    get_coco_stairs_table,
    detect_excluded_rank_columns,
    build_coco_rerun_frame,
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
    info_panel(
        "Workflow: run COCO on the full ranked matrix, inspect `Stairs(2)` for "
        "S1 = n - 1 candidates, rerun COCO on those flagged attributes only, then validate the active result.",
        icon="ℹ️",
    )

    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Analysis Parameters", icon="📐", tight=True)
        job_name = st.text_input(
            "Analysis Name:",
            value="COCO_YO",
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
    
    # Run COCO + exclusion rerun + Validation
    divider()
    
    coco_tables = None
    validation_results = None
    first_pass_tables = st.session_state.get("coco_first_pass_results", {})
    second_pass_tables = st.session_state.get("coco_second_pass_results", {})
    exclusion_candidates = st.session_state.get("coco_exclusion_candidates", [])
    workflow_summary = st.session_state.get("coco_workflow_summary", {})

    if st.button("🚀 Run COCO Exclusion Workflow", type="primary", use_container_width=True):
        workflow = run_coco_exclusion_workflow(ranked_data, job_name, stair_value)
        if workflow and workflow.get("final_tables"):
            first_pass_tables = workflow.get("first_pass_tables") or {}
            second_pass_tables = workflow.get("second_pass_tables") or {}
            exclusion_candidates = workflow.get("exclusion_candidates") or []
            workflow_summary = workflow.get("summary") or {}

            st.session_state.coco_first_pass_results = first_pass_tables
            st.session_state.coco_second_pass_results = second_pass_tables
            st.session_state.coco_exclusion_candidates = exclusion_candidates
            st.session_state.coco_workflow_summary = workflow_summary
            st.session_state.coco_last_job_name = workflow_summary.get("final_job_name", job_name)

            data_manager.store_coco_results(workflow["final_tables"])
            coco_tables = data_manager.get_coco_results()

            if has_validation_requirements(coco_tables):
                validation_results = run_validation_analysis(
                    workflow["active_ranked_data"],
                    coco_tables,
                    job_name=workflow_summary.get("validation_job_name", "StudentRankingInverted"),
                    source_label=workflow_summary.get("final_label", "active COCO result"),
                )
                if validation_results is not None and not validation_results.empty:
                    data_manager.store_validation_results(validation_results)
                    validation_results = data_manager.get_validation_results()
                else:
                    st.warning("Validation did not produce meaningful results. Please review the active COCO output.")
            else:
                info_panel(
                    "Validation skipped because the active COCO output does not include the required columns.",
                    icon="ℹ️",
                )
        else:
            st.error("The COCO workflow did not return any usable tables. Please try again.")
            coco_tables = data_manager.get_coco_results()
            validation_results = data_manager.get_validation_results()
    else:
        coco_tables = data_manager.get_coco_results()
        validation_results = data_manager.get_validation_results()

    if coco_tables:
        display_coco_workflow_summary(workflow_summary, exclusion_candidates)
        divider()

        final_title = "Final COCO Analysis Result"
        if workflow_summary.get("used_second_pass"):
            final_title = "Final COCO Analysis Result (Excluded-Attribute Rerun)"
        display_coco_results(
            coco_tables,
            title=final_title,
            export_file_name="coco_results_final.zip",
        )

        if first_pass_tables and second_pass_tables:
            with st.expander("🔁 Review Initial COCO Pass", expanded=False):
                display_key_tables(first_pass_tables, show_all_expander=False)
                display_export_options(
                    first_pass_tables,
                    section_title="💾 Export Initial COCO Pass",
                    file_name="coco_results_initial_pass.zip",
                    button_label="📦 Download Initial Pass (ZIP)",
                    show_divider=False,
                )

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
        st.info("Validation results will appear automatically after a successful active COCO run.")

    forward_spec = None
    if coco_tables:
        forward_spec = {
            "label": "📊 Explore Visualizations",
            "page": "pages/6_📊_Visualizations.py",
            "key": "pulse1",
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


def build_exclusion_candidates_df(exclusion_candidates):
    if not exclusion_candidates:
        return pd.DataFrame()
    rows = []
    for item in exclusion_candidates:
        rows.append(
            {
                "Attribute": item.get("attribute_label"),
                "Rank Column": item.get("rank_column"),
                "COCO Column": item.get("coco_column"),
                "Stairs(2) S1": item.get("s1_value"),
                "Rule Match": f"S1 = n - 1 = {int(item.get('threshold', 0))}",
            }
        )
    return pd.DataFrame(rows)


def display_coco_workflow_summary(workflow_summary, exclusion_candidates):
    section_header("Exclusion Workflow Summary", icon="🧭")

    if not workflow_summary:
        st.info("Run the COCO exclusion workflow to populate the summary.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Rank Columns", workflow_summary.get("initial_attribute_count", 0))
    with col2:
        st.metric("Flagged Attributes", workflow_summary.get("exclusion_candidate_count", 0))
    with col3:
        st.metric("Active Rank Columns", workflow_summary.get("final_attribute_count", 0))
    with col4:
        final_stage = "Rerun" if workflow_summary.get("used_second_pass") else "Initial Pass"
        st.metric("Active Result", final_stage)

    final_label = workflow_summary.get("final_label", "initial COCO pass")
    if workflow_summary.get("rerun_warning"):
        info_panel(
            "An excluded-attribute rerun was attempted, but the initial COCO pass remains active.",
            icon="⚠️",
        )
    elif workflow_summary.get("used_second_pass"):
        info_panel(
            f"The active COCO result comes from the excluded-attribute rerun. "
            f"Validation is computed against the {final_label.lower()}.",
            icon="✅",
        )
    elif workflow_summary.get("stairs2_found"):
        info_panel(
            f"No rerun was needed. The active COCO result remains the {final_label.lower()}.",
            icon="ℹ️",
        )
    else:
        info_panel(
            "Stairs(2) could not be identified from the COCO output, so the initial pass remains active.",
            icon="⚠️",
        )

    if workflow_summary.get("rerun_warning"):
        st.warning(workflow_summary["rerun_warning"])

    if exclusion_candidates:
        st.markdown("**Flagged attributes from `Stairs(2)`**")
        st.dataframe(build_exclusion_candidates_df(exclusion_candidates), use_container_width=True, hide_index=True)


def run_coco_exclusion_workflow(ranked_data, job_name, stair_value):
    """Run the initial COCO pass, detect exclusion candidates, and rerun on those flagged attributes only."""
    first_pass = run_coco_analysis(
        ranked_data,
        job_name,
        stair_value,
        phase_label="Initial COCO Pass",
        preview_label="Initial Pass",
    )
    if not first_pass or not first_pass.get("tables"):
        return None

    first_pass_tables = first_pass["tables"]
    stairs2_found = not get_coco_stairs_table(first_pass_tables, 2).empty
    exclusion_candidates = detect_excluded_rank_columns(first_pass_tables, ranked_data)
    initial_rank_columns = get_coco_rank_columns(ranked_data)

    final_tables = first_pass_tables
    final_label = "Initial COCO Pass"
    final_job_name = job_name
    validation_job_name = "StudentRankingInverted"
    active_ranked_data = ranked_data
    second_pass_tables = {}
    rerun_warning = None

    if exclusion_candidates:
        rerun_rank_columns = [item["rank_column"] for item in exclusion_candidates]
        rerun_ranked_data = build_coco_rerun_frame(ranked_data, rerun_rank_columns)
        rerun_job_name = f"{job_name}_excluded"

        st.info(
            f"🔁 Rerunning COCO on {len(rerun_rank_columns)} flagged attributes identified from `Stairs(2)`."
        )
        second_pass = run_coco_analysis(
            rerun_ranked_data,
            rerun_job_name,
            stair_value,
            phase_label="Excluded-Attribute COCO Rerun",
            preview_label="Excluded-Attribute Rerun",
        )
        if second_pass and second_pass.get("tables"):
            second_pass_tables = second_pass["tables"]
            final_tables = second_pass_tables
            final_label = "Excluded-Attribute COCO Rerun"
            final_job_name = rerun_job_name
            validation_job_name = "StudentRankingExcludedInverted"
            active_ranked_data = rerun_ranked_data
        else:
            rerun_warning = (
                "The excluded-attribute rerun did not complete successfully. "
                "The initial COCO pass remains the active result."
            )

    summary = {
        "initial_attribute_count": len(initial_rank_columns),
        "exclusion_candidate_count": len(exclusion_candidates),
        "final_attribute_count": len(get_coco_rank_columns(active_ranked_data)),
        "used_second_pass": bool(second_pass_tables),
        "stairs2_found": stairs2_found,
        "final_label": final_label,
        "final_job_name": final_job_name,
        "validation_job_name": validation_job_name,
        "rerun_warning": rerun_warning,
    }

    return {
        "first_pass_tables": first_pass_tables,
        "second_pass_tables": second_pass_tables,
        "final_tables": final_tables,
        "active_ranked_data": active_ranked_data,
        "exclusion_candidates": exclusion_candidates,
        "summary": summary,
    }


def run_coco_analysis(ranked_data, job_name, stair_value, phase_label="COCO Analysis", preview_label="COCO"):
    """Execute a single COCO analysis pass and return parsed tables plus request metadata."""

    with st.spinner(f"Preparing data for {phase_label.lower()}..."):
        matrix_data = prepare_coco_matrix(ranked_data)
        object_names = build_object_names_payload(ranked_data)
    preview_key = str(preview_label).lower().replace(" ", "_").replace("-", "_")

    with st.expander(f"🔍 {preview_label} Matrix Preview", expanded=False):
        st.text_area(
            "COCO Input Matrix (first 500 chars):",
            matrix_data[:500],
            height=150,
            key=f"coco_input_matrix_{preview_key}",
        )
        st.text_area(
            "COCO Object Names (first 500 chars):",
            object_names[:500],
            height=150,
            help="Names are sent in parallel with the matrix so COCO can label objects correctly.",
            key=f"coco_object_names_{preview_key}",
        )

    st.info(f"🌐 Sending {phase_label.lower()} to the COCO service...")

    resp = None
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"Connecting to COCO service for {phase_label.lower()}...")
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

        status_text.text(f"Processing {phase_label.lower()} response...")
        progress_bar.progress(60)

        status_text.text(f"Parsing {phase_label.lower()} results...")
        progress_bar.progress(80)
        tables = parse_coco_html(resp)

        status_text.text(f"Finalizing {phase_label.lower()}...")
        progress_bar.progress(100)

        if not tables:
            handle_coco_error(resp, "❌ COCO analysis returned no tables.")
            return None

        html = decode_coco_response(resp)
        st.session_state.last_coco_html = html
        status_text.text(f"✅ {phase_label} completed!")
        st.success(f"✅ {phase_label} completed!")
        return {
            "tables": tables,
            "html": html,
            "matrix_data": matrix_data,
            "object_names": object_names,
        }

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


def display_coco_results(tables, title="COCO Analysis Result", export_file_name="coco_results.zip"):
    """Render COCO core results and diagnostics."""
    section_header(title, icon="📊")
    st.caption(f"Parsed {len(tables)} tables from the COCO service.")

    display_key_tables(tables)
    display_export_options(tables, file_name=export_file_name)



def display_key_tables(tables, show_all_expander=True):
    """Highlight the key tables returned by COCO."""
    score_table = None
    ranking_table = None

    for name, df in tables.items():
        cols_lower = [str(col).lower() for col in df.columns]
        if any("becsl" in col for col in cols_lower) or any("score" in col for col in cols_lower):
            score_table = (name, df)
        if any("rank" in col for col in cols_lower) or any("rangsor" in col for col in cols_lower):
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

    if show_all_expander:
        with st.expander("🔍 All Result Tables", expanded=False):
            for name, df in tables.items():
                st.markdown(f"**{name}**")
                st.dataframe(clean_coco_dataframe(df), use_container_width=True)


def display_export_options(
    tables,
    section_title="💾 Export Results",
    file_name="coco_results.zip",
    button_label="📦 Download All Results (ZIP)",
    show_divider=True,
):
    """Display options to export COCO results."""
    if show_divider:
        divider()
    st.header(section_title)

    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for table_name, df in tables.items():
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{table_name}.csv", csv_data)

    st.download_button(
        button_label,
        zip_buffer.getvalue(),
        file_name,
        "application/zip",
        use_container_width=True,
    )


def run_validation_analysis(ranked_data, coco_results, job_name="StudentRankingInverted", source_label="active COCO result"):
    """Run the complete validation analysis with progress feedback."""
    main_table = coco_results.get("table_4")
    if main_table is None:
        st.error("Validation requires the COCO output table named 'table_4'.")
        return None

    with st.status(f"🔄 **Running Validation Analysis for {source_label}...**", expanded=True) as status:
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
                job_name=job_name,
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
