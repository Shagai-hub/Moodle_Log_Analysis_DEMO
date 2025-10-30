import streamlit as st
from assets.ui_components import apply_theme, divider, info_panel, page_header, section_header, nav_footer

apply_theme()

page_header(
    "AI Insights",
    "Leverage language models to surface patterns, trends, and anomalies in your Moodle data.",
    icon="🤖",
    align="left",
    compact=True,
)

info_panel(
    "This workspace will host conversational AI helpers, automated summaries, and predictive flags.",
    icon="💡",
)

divider()
st.info("🚧 This experience is being prepared. Check back soon for AI-powered insights!")

divider()
nav_footer(
    back={
        "label": "⬅️ Back to Validation",
        "page": "pages/6_✅_Validation.py",
        "key": "nav_back_to_validation_from_ai",
        "fallback": "✅ Validation",
    }
)
