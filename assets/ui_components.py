"""
Shared UI helpers for the Moodle Log Analyzer Streamlit app.

This module centralises theme loading and small UI patterns so pages
can stay lean and consistent while keeping business logic untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

_ASSETS_ROOT = Path(__file__).resolve().parent
_THEME_KEY = "theme_css_loaded"


def apply_theme() -> None:
    """Inject the shared CSS theme."""
    css_path = _ASSETS_ROOT / "styles.css"
    if not css_path.exists():
        return

    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_THEME_KEY] = True


def page_header(
    title: str,
    subtitle: Optional[str] = None,
    *,
    icon: Optional[str] = None,
    kicker: Optional[str] = None,
    align: str = "center",
    compact: bool = False,
) -> None:
    """Render a consistent hero/header block."""
    classes = ["page-header"]
    if align == "left":
        classes.append("page-header--left")
    if compact:
        classes.append("page-header--compact")

    emoji_html = f"<span class='page-header__emoji'>{icon}</span>" if icon else ""
    kicker_html = f"<span class='page-header__kicker'>{kicker}</span>" if kicker else ""
    subtitle_html = f"<p class='page-header__subtitle'>{subtitle}</p>" if subtitle else ""

    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
          {kicker_html}
          <h1 class="page-header__title">{emoji_html}{title}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    subtitle: Optional[str] = None,
    *,
    icon: Optional[str] = None,
    tight: bool = False,
) -> None:
    """Render a section headline with the branded dot accent."""
    classes = ["section-title"]
    if tight:
        classes.append("section-title--tight")

    text = f"{icon} {title}" if icon else title
    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
          <span class="dot"></span>
          <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(f"<p class='section-caption'>{subtitle}</p>", unsafe_allow_html=True)


def info_panel(
    body: str,
    *,
    title: Optional[str] = None,
    icon: Optional[str] = None,
    subtle: bool = False,
) -> None:
    """Render a text-focused information panel with optional icon/title."""
    classes = ["panel"]
    if subtle:
        classes.append("panel--subtle")

    icon_html = f"<div class='panel__icon'>{icon}</div>" if icon else ""
    title_html = f"<div class='panel__title'>{title}</div>" if title else ""

    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
          {icon_html}
          {title_html}
          <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Streamlit-friendly horizontal rule."""
    st.markdown("<hr>", unsafe_allow_html=True)


def subtle_text(text: str) -> None:
    """Render supporting text styled like a caption."""
    st.markdown(f"<p class='section-caption'>{text}</p>", unsafe_allow_html=True)
