import streamlit as st
import pandas as pd
import altair as alt
from utils.session_data_manager import SessionDataManager
from utils.coco_utils import send_coco_request, parse_coco_html, invert_ranking

# Safe initialization
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()

def main():
    st.title("✅ Validation")
    st.markdown("Validate COCO analysis results by checking inverted rankings.")

    data_manager = st.session_state.data_manager

    # Load ranked data and COCO results
    ranked_data = data_manager.get_ranked_results()
    coco_table4 = data_manager.get_coco_table("table_4")

    if ranked_data is None or coco_table4 is None:
        st.warning("📊 Please run COCO analysis first to enable validation.")
        return

    st.success("✅ Ranked data and COCO results loaded successfully.")

    if st.button("🚀 Run Validation", use_container_width=True):
        st.info("Running validation...")

        try:
            # Step 1: Extract numeric values for inversion
            matrix_df = ranked_data.drop(columns=["userid", "userfullname"], errors="ignore")

            # Step 2: Invert the ranking
            inverted_matrix_df = invert_ranking(matrix_df)

            # Step 3: Convert inverted matrix to string format for COCO
            inverted_matrix_lines = []
            for _, row in inverted_matrix_df.iterrows():
                row_str = "\t".join(str(val) for val in row)
                inverted_matrix_lines.append(row_str)

            inverted_matrix_data = "\n".join(inverted_matrix_lines)

            # Step 4: Send inverted matrix to COCO
            html_response_inverted = send_coco_request(
                inverted_matrix_data,
                job_name="StudentRankingInverted",
                stair=str(len(matrix_df)),
                object_names="",
                attribute_names="",
                keep_files=False
            )

            # Step 5: Parse the inverted COCO response to get all tables
            tables_inverted = parse_coco_html(html_response_inverted)

            # Step 6: Perform validation
            if "table_4" in tables_inverted:
                df_table4_inverted = tables_inverted["table_4"]

                # Find Delta_T_ny columns in both tables
                delta_col_original = next(
                    (col for col in coco_table4.columns if "Delta/TÃ©ny" in col.lower()), None
                )
                delta_col_inverted = next(
                    (col for col in df_table4_inverted.columns if "Delta/TÃ©ny" in col.lower()), None
                )

                if delta_col_original and delta_col_inverted:
                    # Convert to numeric
                    original_delta = pd.to_numeric(coco_table4[delta_col_original], errors="coerce")
                    inverted_delta = pd.to_numeric(df_table4_inverted[delta_col_inverted], errors="coerce")

                    # Calculate validation: original_delta * inverted_delta <= 0 is valid
                    validation_product = original_delta * inverted_delta
                    is_valid = validation_product <= 0

                    # Step 7: Add results to ranked table
                    ranked_validated = ranked_data.copy()
                    ranked_validated["Validation_Result"] = is_valid.map({True: "Valid", False: "Invalid"})

                    # Store validated results
                    data_manager.store_validated_results(ranked_validated)

                    st.success("Validation completed! Results stored successfully.")

                    # Show validation summary
                    valid_count = is_valid.sum()
                    total_count = len(is_valid)
                    st.write(f"Valid results: {valid_count}/{total_count} ({valid_count / total_count * 100:.1f}%)")

                    # Display validated results
                    st.subheader("Validated Results")
                    st.dataframe(ranked_validated)

                    # Build bar chart with color coding
                    chart_df = ranked_validated[["userfullname", "Validation_Result"]].copy()
                    bar_chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("userfullname:N", sort=None, title="Student"),
                            color=alt.Color(
                                "Validation_Result:N",
                                scale=alt.Scale(domain=["Valid", "Invalid"], range=["green", "red"])
                            )
                        )
                        .properties(width=700, height=400, title="Validation Results")
                    )
                    st.altair_chart(bar_chart, use_container_width=True)

                    # Download button for validated results
                    csv_validated = ranked_validated.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download Validated Results (CSV)",
                        csv_validated,
                        "validated_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Delta columns not found in COCO tables.")
            else:
                st.error("Inverted table_4 not found in COCO results.")
        except Exception as e:
            st.error(f"An error occurred during validation: {e}")

if __name__ == "__main__":
    main()