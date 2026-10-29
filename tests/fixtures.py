"""
tests/fixtures.py

Вспомогательные функции и фейковые (fake) реализации тяжёлых зависимостей
(embeddings, ChromaDB, Ollama), чтобы юнит-тесты не требовали скачивания
моделей или запущенного Ollama.
"""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from core.schemas import RetrievedChunk


def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """Генерирует простой синтетический PDF с заданным текстом по страницам."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages_text:
        y = 750
        for line in text.splitlines() or [text]:
            c.drawString(72, y, line[:100])
            y -= 18
        c.showPage()
    c.save()
    return buf.getvalue()


class FakeEmbeddingModel:
    """Детерминированная псевдо-эмбеддинг модель без тяжёлых зависимостей.

    Использует простой мешок символов (bag-of-chars), чтобы похожие
    тексты давали более близкие векторы — этого достаточно для unit-тестов
    логики порога/сортировки, не для оценки реального качества retrieval.
    """

    DIM = 32

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for ch in text.lower():
            vec[ord(ch) % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, question: str) -> list[float]:
        return self._vectorize(question)


class FakeOllamaClient:
    """Фейковый Ollama-клиент: возвращает заранее заданный текст."""

    def __init__(self, response_text: str = "", available: bool = True, raise_error: bool = False):
        self.response_text = response_text
        self.available = available
        self.raise_error = raise_error
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def generate(self, prompt: str, system: str | None = None, temperature=None, timeout=None):
        from llm.ollama_client import GenerationResult, OllamaUnavailableError

        self.calls.append(prompt)
        if self.raise_error:
            raise OllamaUnavailableError("Ollama недоступна (тестовая заглушка).")
        return GenerationResult(text=self.response_text, model="fake-model")


def make_retrieved_chunk(
    source_id: str = "S1",
    text: str = "Пример текста чанка.",
    source_name: str = "Lecture_01.pdf",
    page: int | None = 12,
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{source_id}",
        text=text,
        source_name=source_name,
        source_type="pdf",
        page=page,
        section=None,
        score=score,
        source_id=source_id,
    )
