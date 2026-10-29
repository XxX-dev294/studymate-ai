"""
ui/sidebar.py

Боковая панель: загрузка файлов, кнопка индексации, прогресс,
статистика индекса, панель состояния Ollama. UI не вызывает
Chroma/Ollama напрямую — только через сервисный слой.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from ingestion.loader import UploadedFileLike
from services.indexing_service import IndexingService
from ui.components.source_card import render_status_badge


def render_sidebar(indexing_service: IndexingService, ollama_available: bool) -> None:
    with st.sidebar:
        st.header(settings.app_name)
        st.caption("Локальный помощник для поиска по учебным материалам")

        st.subheader("Статус системы")
        render_status_badge(ollama_available, f"Ollama ({settings.llm_model})")
        overview = indexing_service.get_index_overview()
        st.caption(f"Чанков в индексе: {overview['chunks_total']}")
        if not ollama_available:
            st.info(
                "Чтобы запустить Ollama: выполните `ollama serve` в терминале, "
                f"затем `ollama pull {settings.llm_model}`."
            )

        st.subheader("Загрузка материалов")
        uploaded = st.file_uploader(
            "Выберите PDF, TXT или MD файлы",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help=f"Лимит: {settings.upload_max_mb} МБ на файл.",
        )

        index_clicked = st.button(
            "Индексировать", type="primary", disabled=not uploaded
        )

        if index_clicked and uploaded:
            files = [UploadedFileLike.from_streamlit(f) for f in uploaded]
            with st.spinner("Обрабатываю документы и строю индекс…"):
                stats = indexing_service.index_files(files)
            st.session_state["last_index_stats"] = stats

        stats = st.session_state.get("last_index_stats")
        if stats is not None:
            st.markdown("**Результат индексации**")
            st.write(
                f"Файлов: {stats.files_total} · "
                f"Проиндексировано: {stats.files_indexed} · "
                f"Пропущено: {stats.files_skipped} · "
                f"Чанков создано: {stats.chunks_created}"
            )
            if stats.errors:
                with st.expander(f"Предупреждения/ошибки ({len(stats.errors)})"):
                    for err in stats.errors:
                        st.warning(err)
