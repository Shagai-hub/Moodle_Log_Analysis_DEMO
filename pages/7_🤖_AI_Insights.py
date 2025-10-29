import streamlit as st
from assets.ui_components import apply_theme, divider, info_panel, page_header

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
