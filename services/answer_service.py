"""
services/answer_service.py

Контракт: AnswerService.answer(question) -> AnswerResult

Реализует RAG-пайплайн и защиту от галлюцинаций:
1. Валидация пустого вопроса.
2. Retrieval top-k с similarity gate.
3. Если нет контекста выше порога -> фиксированный отказ (no_context/
   index_empty), LLM не вызывается.
4. Промпт с [S1..Sn], вызов Ollama.
5. Валидация ссылок [Sx] относительно реально найденных чанков; при
   недопустимых ссылках — один повтор, затем безопасный отказ.
6. Источники для UI строятся из метаданных retrieval, а не из текста LLM.
"""

from __future__ import annotations

import re

from config import settings
from core.schemas import (
    STATUS_ERROR,
    STATUS_INDEX_EMPTY,
    STATUS_INVALID_CITATIONS,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_CONTEXT,
    STATUS_OK,
    AnswerResult,
    RetrievedChunk,
    SourceRef,
)
from llm.ollama_client import OllamaClient, OllamaUnavailableError
from llm.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt, build_retry_prompt
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

NO_CONTEXT_MESSAGE = "В загруженных материалах информация по данному вопросу не найдена."
INVALID_CITATIONS_MESSAGE = (
    "Не удалось сформировать ответ с проверенными ссылками на источники. "
    "Попробуйте переформулировать вопрос."
)

_CITATION_RE = re.compile(r"\[S(\d+)\]")


def _extract_citation_ids(text: str) -> set[str]:
    return {f"S{m}" for m in _CITATION_RE.findall(text)}


def _locator_for(chunk: RetrievedChunk) -> str:
    if chunk.page is not None:
        return f"стр. {chunk.page}"
    if chunk.section:
        return f"раздел «{chunk.section}»"
    return "фрагмент"


def _build_source_refs(chunks: list[RetrievedChunk], limit: int) -> list[SourceRef]:
    refs = []
    for c in chunks[:limit]:
        snippet = c.text[:280] + ("…" if len(c.text) > 280 else "")
        refs.append(
            SourceRef(
                source_id=c.source_id,
                source_name=c.source_name,
                locator=_locator_for(c),
                snippet=snippet,
                chunk_id=c.chunk_id,
                score=c.score,
            )
        )
    return refs


class AnswerService:
    """Оркестрирует вопрос -> retrieval -> LLM -> проверенный ответ."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        ollama_client: OllamaClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.ollama_client = ollama_client or OllamaClient()
        self.vector_store = vector_store or (self.retriever.vector_store)

    def answer(self, question: str) -> AnswerResult:
        question = (question or "").strip()
        if not question:
            return AnswerResult(
                status=STATUS_ERROR,
                answer_text="Вопрос не может быть пустым.",
                reason="empty_question",
            )

        if self.vector_store.count() == 0:
            return AnswerResult(
                status=STATUS_INDEX_EMPTY,
                answer_text=(
                    "Индекс пуст. Сначала загрузите и проиндексируйте учебные "
                    "материалы на экране «Индексация»."
                ),
                reason="index_empty",
            )

        chunks = self.retriever.search(question)
        if not chunks:
            return AnswerResult(
                status=STATUS_NO_CONTEXT,
                answer_text=NO_CONTEXT_MESSAGE,
                reason="no_relevant_context",
                retrieval_scores=[],
            )

        scores = [c.score for c in chunks]
        valid_ids = {c.source_id for c in chunks}

        try:
            prompt = build_answer_prompt(question, chunks)
            result = self.ollama_client.generate(
                prompt=prompt,
                system=ANSWER_SYSTEM_PROMPT,
                temperature=settings.llm_temperature,
            )
        except OllamaUnavailableError as exc:
            return AnswerResult(
                status=STATUS_LLM_UNAVAILABLE,
                answer_text=(
                    "Локальная модель Ollama недоступна. Запустите 'ollama serve' "
                    "и убедитесь, что нужная модель загружена."
                ),
                reason=str(exc),
                retrieval_scores=scores,
            )

        answer_text = result.text.strip()
        used_ids = _extract_citation_ids(answer_text)
        invalid_ids = used_ids - valid_ids

        if invalid_ids:
            # Одна попытка повторной генерации с валидными ID.
            try:
                retry_prompt = build_retry_prompt(question, chunks, sorted(invalid_ids))
                retry_result = self.ollama_client.generate(
                    prompt=retry_prompt,
                    system=ANSWER_SYSTEM_PROMPT,
                    temperature=settings.llm_temperature,
                )
            except OllamaUnavailableError as exc:
                return AnswerResult(
                    status=STATUS_LLM_UNAVAILABLE,
                    answer_text=(
                        "Локальная модель Ollama недоступна при повторной попытке."
                    ),
                    reason=str(exc),
                    retrieval_scores=scores,
                )

            answer_text = retry_result.text.strip()
            used_ids = _extract_citation_ids(answer_text)
            invalid_ids = used_ids - valid_ids
            if invalid_ids:
                return AnswerResult(
                    status=STATUS_INVALID_CITATIONS,
                    answer_text=INVALID_CITATIONS_MESSAGE,
                    reason=f"invalid_citation_ids:{sorted(invalid_ids)}",
                    retrieval_scores=scores,
                )

        # Источники строятся сервером из метаданных retrieval — только те,
        # что реально использовала модель (used_ids), либо, если модель
        # не сослалась ни на один источник явно, показываем топ найденных.
        cited_chunks = [c for c in chunks if c.source_id in used_ids] or chunks
        sources = _build_source_refs(cited_chunks, settings.max_sources_in_answer)

        return AnswerResult(
            status=STATUS_OK,
            answer_text=answer_text,
            sources=sources,
            retrieval_scores=scores,
        )
