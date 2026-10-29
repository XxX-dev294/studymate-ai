"""
tests/test_answer_guard.py

Проверяет защиту от галлюцинаций и RAG-пайплайн
через AnswerService, с фейковыми retriever/ollama (без сети и без
реального Ollama).
"""

from __future__ import annotations

import unittest

from core.schemas import (
    STATUS_ERROR,
    STATUS_INDEX_EMPTY,
    STATUS_INVALID_CITATIONS,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_CONTEXT,
    STATUS_OK,
)
from services.answer_service import AnswerService
from tests.fixtures import FakeOllamaClient, make_retrieved_chunk


class FakeVectorStoreCount:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakeRetriever:
    def __init__(self, chunks, total_count: int = 1) -> None:
        self._chunks = chunks
        self.vector_store = FakeVectorStoreCount(total_count)

    def search(self, question, top_k=None, filters=None, min_similarity=None):
        return self._chunks


class TestAnswerServiceGuards(unittest.TestCase):
    def test_empty_question_returns_error_status(self) -> None:
        service = AnswerService(
            retriever=FakeRetriever(chunks=[], total_count=1),
            ollama_client=FakeOllamaClient(),
        )
        result = service.answer("   ")
        self.assertEqual(result.status, STATUS_ERROR)

    def test_empty_index_returns_index_empty_status_without_calling_llm(self) -> None:
        ollama = FakeOllamaClient(response_text="не должно вызваться")
        service = AnswerService(
            retriever=FakeRetriever(chunks=[], total_count=0),
            ollama_client=ollama,
        )
        result = service.answer("Что такое SWOT-анализ?")
        self.assertEqual(result.status, STATUS_INDEX_EMPTY)
        self.assertEqual(ollama.calls, [])

    def test_no_relevant_context_returns_fixed_refusal_without_calling_llm(self) -> None:
        ollama = FakeOllamaClient(response_text="не должно вызваться")
        service = AnswerService(
            retriever=FakeRetriever(chunks=[], total_count=5),
            ollama_client=ollama,
        )
        result = service.answer("Вопрос вне контекста материалов")
        self.assertEqual(result.status, STATUS_NO_CONTEXT)
        self.assertIn("не найдена", result.answer_text)
        self.assertEqual(ollama.calls, [])

    def test_ok_answer_includes_sources_from_retrieval_metadata(self) -> None:
        chunk = make_retrieved_chunk(source_id="S1", source_name="Lecture_01.pdf", page=12)
        ollama = FakeOllamaClient(response_text="SWOT-анализ — это метод оценки [S1].")
        service = AnswerService(
            retriever=FakeRetriever(chunks=[chunk], total_count=5),
            ollama_client=ollama,
        )
        result = service.answer("Что такое SWOT-анализ?")
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].source_name, "Lecture_01.pdf")
        self.assertEqual(result.sources[0].locator, "стр. 12")

    def test_llm_unavailable_status_when_ollama_down(self) -> None:
        chunk = make_retrieved_chunk()
        ollama = FakeOllamaClient(raise_error=True)
        service = AnswerService(
            retriever=FakeRetriever(chunks=[chunk], total_count=5),
            ollama_client=ollama,
        )
        result = service.answer("Вопрос")
        self.assertEqual(result.status, STATUS_LLM_UNAVAILABLE)

    def test_invalid_citation_triggers_retry_then_safe_refusal(self) -> None:
        chunk = make_retrieved_chunk(source_id="S1")
        # Модель всегда ссылается на несуществующий источник S99, даже после повтора.
        ollama = FakeOllamaClient(response_text="Ответ со ссылкой на [S99].")
        service = AnswerService(
            retriever=FakeRetriever(chunks=[chunk], total_count=5),
            ollama_client=ollama,
        )
        result = service.answer("Вопрос")
        self.assertEqual(result.status, STATUS_INVALID_CITATIONS)
        # Должна быть ровно одна повторная попытка -> 2 вызова LLM.
        self.assertEqual(len(ollama.calls), 2)

    def test_invalid_citation_recovers_on_retry(self) -> None:
        chunk = make_retrieved_chunk(source_id="S1")

        class RetryingOllama(FakeOllamaClient):
            def generate(self, prompt, system=None, temperature=None, timeout=None):
                from llm.ollama_client import GenerationResult

                self.calls.append(prompt)
                if len(self.calls) == 1:
                    return GenerationResult(text="Неверная ссылка [S99].", model="fake")
                return GenerationResult(text="Исправленный ответ [S1].", model="fake")

        ollama = RetryingOllama()
        service = AnswerService(
            retriever=FakeRetriever(chunks=[chunk], total_count=5),
            ollama_client=ollama,
        )
        result = service.answer("Вопрос")
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(len(ollama.calls), 2)


if __name__ == "__main__":
    unittest.main()
