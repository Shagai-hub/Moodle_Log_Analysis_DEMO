import streamlit as st
from pathlib import Path

from assets.ui_components import apply_theme, divider, info_panel, nav_footer, page_header, section_header

apply_theme()
ISSUE_URL = "https://github.com/Shagai-hub/Moodle_Log_Analysis_DEMO/issues/new"


def _render_quick_start() -> None:
    section_header("Quick Start", tight=True)
    st.markdown(
        """
1. Open **Data Upload** and upload a Moodle `CSV` or `XLSX` export.
2. Go to **Configuration** and set professor names plus exam deadlines.
3. Open **Attribute Analysis**, select metrics, and click **Compute Attributes**.
4. Continue to **Ranking** and run student ranking.
5. Run **COCO Analysis** and review **Visualizations** and **AI Insights**.
        """
    )


def _render_common_issues() -> None:
    section_header("Common Issues", icon="")

    with st.expander("Upload fails with missing columns"):
        st.markdown(
            """
- Required columns are `userid`, `userfullname`, and `message`.
- Fix column names in the export file, then upload again.
            """
        )

    with st.expander("Excel file fails to load"):
        st.markdown(
            """
- The environment may be missing `openpyxl`.
- Install it with `pip install openpyxl` or upload CSV instead.
            """
        )

    with st.expander("Compute Attributes is slow or returns many zeros"):
        st.markdown(
            """
- Some attributes need extra columns (`created`, `parent`, `subject`) or NLP packages.
- Deselect unsupported attributes or install missing dependencies.
            """
        )

    with st.expander("Ranking, COCO, or Visualizations show missing data warnings"):
        st.markdown(
            """
- These pages require previous steps to be completed in order.
- Re-run Attribute Analysis, then Ranking, then COCO to refresh session data.
            """
        )

    with st.expander("AI Insights shows missing-columns warnings"):
        st.markdown(
            """
- Rules may reference fields not present in your computed dataset.
- Re-check configured rules and ensure attribute names match exactly.
            """
        )


def _render_user_guide_download() -> None:
    section_header("User Guide", icon="")
    guide_path = Path("USER_GUIDE 2025.docx")
    if guide_path.exists():
        with guide_path.open("rb") as f:
            st.download_button(
                "📥 Download Full User Guide (DOCX)",
                data=f.read(),
                file_name=guide_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="download_user_guide_btn",
            )
    else:
        st.warning("User guide file not found at `USER_GUIDE 2025.docx`.")


def main() -> None:
    page_header(
        "Help Center",
        "Simple guidance for using the Moodle Log Analyzer end-to-end.",
        icon="❓",
        align="left",
        compact=True,
    )


    divider()
    _render_quick_start()

    section_header("Workflow Notes", icon="")
    st.markdown(
        """
- Start from `Data Upload`, then continue page-by-page in order.
- If a page shows missing prerequisites, return to the previous step and re-run it.
- Export outputs after key stages for reproducibility (`Configuration`, `OAM`, `Ranking`, and `COCO` tables).
        """
    )

    divider()
    _render_common_issues()

    divider()
    _render_user_guide_download()

    divider()
    section_header("Need More Help?", icon="✉️")
    st.markdown(
        """
- Capture a screenshot of the page and the error message.
- Note which step you were on (`Upload`, `Configuration`, `Attribute Analysis`, etc.).
- Use the issue tracker to report problems: [Report an Issue]({ISSUE_URL}).
        """
        .format(ISSUE_URL="mailto:ssun41268@gmail.com?subject=Report%20an%20issue")
    )

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to Home",
            "page": "Analysis.py",
            "key": "nav_back_home_from_help",
            "fallback": "Home",
        },
        message="This help page summarizes the official guide for quick in-app support.",
        forward={
            "label": "➡️ Go to Data Upload",
            "page": "pages/1_📊_Data_Upload.py",
            "key": "nav_to_upload_from_help",
            "fallback": "📊 Data Upload",
        },
    )


if __name__ == "__main__":
    main()
