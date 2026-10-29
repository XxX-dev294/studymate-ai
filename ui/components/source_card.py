"""
ui/components/source_card.py

Отрисовка плашек источников. Источник — метаданные реального найденного
чанка (SourceRef), не придуманный текст LLM.
"""

from __future__ import annotations

import streamlit as st

from core.schemas import SourceRef


def render_sources(sources: list[SourceRef]) -> None:
    """Рисует блок источников с раскрываемым фрагментом текста."""
    if not sources:
        return

    st.markdown("**Источники:**")
    for src in sources:
        score_text = f" · релевантность {src.score:.2f}" if src.score is not None else ""
        label = f"[{src.source_id}] {src.source_name} — {src.locator}{score_text}"
        with st.expander(label):
            st.caption("Показать подтверждение (фрагмент из документа):")
            st.write(src.snippet)


def render_status_badge(available: bool, label: str) -> None:
    if available:
        st.success(f"✅ {label}")
    else:
        st.error(f"❌ {label}")
