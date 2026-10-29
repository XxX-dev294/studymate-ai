"""
services/summary_service.py

Контракт:
SummaryService.summarize(topic=None, document_id=None, style="brief")
    -> AnswerResult (используется та же структура, что и для ответа)

Режим конспекта:
- Вход: выбранный файл (document_id) или тема (topic).
- Для темы — retrieval по материалам; для файла — все чанки документа,
  при превышении лимита контекста — суммирование по частям.
- Конспект обязан указывать источники; не дополнять внешними знаниями.
"""

from __future__ import annotations

from config import settings
from core.schemas import (
    STATUS_ERROR,
    STATUS_INDEX_EMPTY,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_CONTEXT,
    STATUS_OK,
    AnswerResult,
    RetrievedChunk,
)
from llm.ollama_client import OllamaClient, OllamaUnavailableError
from llm.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore
from services.answer_service import _build_source_refs, _extract_citation_ids


def _chunks_from_raw_get(raw: dict) -> list[RetrievedChunk]:
    """Преобразует результат VectorStore.get_chunks_by_document_id в RetrievedChunk."""
    ids = raw.get("ids", [])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    # Сортируем по page, затем chunk_index, чтобы конспект шёл по порядку документа.
    items = list(zip(ids, documents, metadatas))
    items.sort(
        key=lambda t: (
            (t[2] or {}).get("page") if (t[2] or {}).get("page") not in (None, -1) else 0,
            (t[2] or {}).get("chunk_index", 0),
        )
    )

    chunks = []
    for i, (chunk_id, text, meta) in enumerate(items):
        meta = meta or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source_name=meta.get("source_name", "unknown"),
                source_type=meta.get("source_type", "unknown"),
                page=VectorStore.page_from_meta(meta.get("page")),
                section=VectorStore.section_from_meta(meta.get("section")),
                score=1.0,
                source_id=f"S{i + 1}",
            )
        )
    return chunks


def _fit_to_context(chunks: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """Берёт чанки по порядку, пока не достигнут лимит символов контекста."""
    out = []
    total = 0
    for c in chunks:
        if total + len(c.text) > max_chars and out:
            break
        out.append(c)
        total += len(c.text)
    return out


class SummaryService:
    """Создаёт краткий конспект по теме (retrieval) или по выбранному файлу."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        ollama_client: OllamaClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.ollama_client = ollama_client or OllamaClient()
        self.vector_store = vector_store or self.retriever.vector_store

    def summarize(
        self,
        topic: str | None = None,
        document_id: str | None = None,
        style: str = "brief",
    ) -> AnswerResult:
        if not topic and not document_id:
            return AnswerResult(
                status=STATUS_ERROR,
                answer_text="Укажите тему или выберите файл для конспекта.",
                reason="missing_input",
            )

        if self.vector_store.count() == 0:
            return AnswerResult(
                status=STATUS_INDEX_EMPTY,
                answer_text="Индекс пуст. Сначала загрузите и проиндексируйте материалы.",
                reason="index_empty",
            )

        truncated = False
        if document_id:
            raw = self.vector_store.get_chunks_by_document_id(document_id)
            all_chunks = _chunks_from_raw_get(raw)
            if not all_chunks:
                return AnswerResult(
                    status=STATUS_NO_CONTEXT,
                    answer_text="Выбранный документ не найден в индексе.",
                    reason="document_not_found",
                )
            chunks = _fit_to_context(all_chunks, settings.max_context_chars)
            truncated = len(chunks) < len(all_chunks)
            topic_or_file = chunks[0].source_name
        else:
            chunks = self.retriever.search(topic, top_k=max(settings.top_k, 6))
            if not chunks:
                return AnswerResult(
                    status=STATUS_NO_CONTEXT,
                    answer_text=(
                        "По указанной теме не найдено материалов в загруженных "
                        "документах."
                    ),
                    reason="no_relevant_context",
                )
            chunks = _fit_to_context(chunks, settings.max_context_chars)
            topic_or_file = topic

        max_points = settings.summary_max_points

        try:
            prompt = build_summary_prompt(topic_or_file, chunks, max_points)
            result = self.ollama_client.generate(
                prompt=prompt,
                system=SUMMARY_SYSTEM_PROMPT,
                temperature=settings.llm_summary_temperature,
            )
        except OllamaUnavailableError as exc:
            return AnswerResult(
                status=STATUS_LLM_UNAVAILABLE,
                answer_text="Локальная модель Ollama недоступна для создания конспекта.",
                reason=str(exc),
            )

        answer_text = result.text.strip()
        if truncated:
            answer_text += (
                "\n\n_Материал превышает лимит контекста — конспект составлен по "
                "первой части документа, а не по всему файлу целиком._"
            )

        used_ids = _extract_citation_ids(answer_text)
        cited_chunks = [c for c in chunks if c.source_id in used_ids] or chunks
        sources = _build_source_refs(cited_chunks, limit=max(settings.max_sources_in_answer, 6))

        return AnswerResult(status=STATUS_OK, answer_text=answer_text, sources=sources)
