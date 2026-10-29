"""
app.py

Точка входа: `streamlit run app.py`.
Собирает интерфейс из ui/*, вызывает только сервисный слой (services/*),
не обращается к Chroma/Ollama напрямую.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from core.schemas import (
    STATUS_ERROR,
    STATUS_INDEX_EMPTY,
    STATUS_INVALID_CITATIONS,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_CONTEXT,
    STATUS_OK,
    AnswerResult,
)
from llm.ollama_client import OllamaClient
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore
from services.answer_service import AnswerService
from services.indexing_service import IndexingService
from services.summary_service import SummaryService
from ui.components.source_card import render_sources
from ui.sidebar import render_sidebar

st.set_page_config(page_title=settings.app_name, page_icon="📚", layout="wide")


@st.cache_resource
def get_services():
    """Создаёт сервисы один раз на сессию сервера (модели кешируются)."""
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store)
    ollama_client = OllamaClient()
    indexing_service = IndexingService(vector_store=vector_store)
    answer_service = AnswerService(retriever=retriever, ollama_client=ollama_client)
    summary_service = SummaryService(retriever=retriever, ollama_client=ollama_client)
    return {
        "vector_store": vector_store,
        "indexing_service": indexing_service,
        "answer_service": answer_service,
        "summary_service": summary_service,
        "ollama_client": ollama_client,
    }


def _render_answer_result(result: AnswerResult) -> None:
    if result.status == STATUS_OK:
        st.markdown(result.answer_text)
        render_sources(result.sources)
    elif result.status in (STATUS_NO_CONTEXT, STATUS_INDEX_EMPTY):
        st.warning(result.answer_text)
        if result.status == STATUS_NO_CONTEXT:
            st.caption(
                "Попробуйте загрузить документы по этой теме на экране «Индексация»."
            )
    elif result.status == STATUS_LLM_UNAVAILABLE:
        st.error(result.answer_text)
    elif result.status == STATUS_INVALID_CITATIONS:
        st.error(result.answer_text)
    elif result.status == STATUS_ERROR:
        st.error(result.answer_text)
    else:  # защитный код на случай нового статуса
        st.error("Непредвиденный статус ответа.")


def render_chat_tab(answer_service: AnswerService) -> None:
    st.subheader("💬 Чат по материалам")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, content in st.session_state["chat_history"]:
        with st.chat_message(role):
            if isinstance(content, AnswerResult):
                _render_answer_result(content)
            else:
                st.markdown(content)

    question = st.chat_input("Задайте вопрос по загруженным материалам…")
    if question:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Ищу материалы и формирую ответ…"):
                result = answer_service.answer(question)
            _render_answer_result(result)
        st.session_state["chat_history"].append(("assistant", result))


def render_summary_tab(summary_service: SummaryService, vector_store: VectorStore) -> None:
    st.subheader("📝 Конспект / шпаргалка")

    mode = st.radio(
        "Источник конспекта", ["По теме", "По файлу"], horizontal=True, key="summary_mode"
    )

    if mode == "По теме":
        topic = st.text_input("Тема для конспекта", key="summary_topic")
        if st.button("Создать конспект", key="summary_by_topic_btn"):
            if not topic.strip():
                st.warning("Введите тему.")
            else:
                with st.spinner("Собираю материалы и формирую конспект…"):
                    result = summary_service.summarize(topic=topic)
                _render_answer_result(result)
    else:
        docs = vector_store.list_documents()
        if not docs:
            st.info("Сначала проиндексируйте хотя бы один документ.")
            return
        options = {f"{d['source_name']}": d["document_id"] for d in docs}
        choice = st.selectbox("Выберите файл", list(options.keys()), key="summary_file")
        if st.button("Создать конспект", key="summary_by_file_btn"):
            with st.spinner("Собираю материалы и формирую конспект…"):
                result = summary_service.summarize(document_id=options[choice])
            _render_answer_result(result)


def main() -> None:
    services = get_services()

    ollama_available = services["ollama_client"].is_available()
    render_sidebar(services["indexing_service"], ollama_available)

    st.title(f"📚 {settings.app_name}")
    st.caption(
        "Локальный ИИ-помощник для поиска, объяснения и конспектирования "
        "учебных материалов. Ответы ограничены вашими загруженными файлами."
    )

    tab_chat, tab_summary = st.tabs(["Чат", "Конспект"])
    with tab_chat:
        render_chat_tab(services["answer_service"])
    with tab_summary:
        render_summary_tab(services["summary_service"], services["vector_store"])


if __name__ == "__main__":
    main()
