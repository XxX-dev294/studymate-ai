"""
core/schemas.py

Единые типы данных, которыми обмениваются все модули StudyMate AI.
От этих контрактов зависят все остальные модули, поэтому менять их стоит осторожно.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# Допустимые статусы результата ответа/конспекта.
STATUS_OK = "ok"
STATUS_NO_CONTEXT = "no_context"
STATUS_INDEX_EMPTY = "index_empty"
STATUS_LLM_UNAVAILABLE = "llm_unavailable"
STATUS_INVALID_CITATIONS = "invalid_citations"
STATUS_ERROR = "error"

VALID_STATUSES = {
    STATUS_OK,
    STATUS_NO_CONTEXT,
    STATUS_INDEX_EMPTY,
    STATUS_LLM_UNAVAILABLE,
    STATUS_INVALID_CITATIONS,
    STATUS_ERROR,
}

SOURCE_TYPE_PDF = "pdf"
SOURCE_TYPE_TXT = "txt"
SOURCE_TYPE_MD = "md"
VALID_SOURCE_TYPES = {SOURCE_TYPE_PDF, SOURCE_TYPE_TXT, SOURCE_TYPE_MD}


def utc_now_iso() -> str:
    """Текущее время в ISO 8601 UTC — используется как дата индексации."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRecord:
    """Текстовый элемент после парсинга, до чанкинга.

    Для PDF, как правило, один record на страницу.
    """

    document_id: str
    source_name: str
    source_type: str  # "pdf" | "txt" | "md"
    text: str
    page: int | None = None          # 1-based номер страницы PDF
    section: str | None = None       # заголовок секции для MD
    line_start: int | None = None    # 1-based начало диапазона строк (TXT)
    line_end: int | None = None      # 1-based конец диапазона строк (TXT)
    content_hash: str = ""


@dataclass
class DocumentChunk:
    """Фрагмент, готовый к созданию embedding и сохранению в Chroma."""

    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    text: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    content_hash: str = ""
    indexed_at: str = field(default_factory=utc_now_iso)


@dataclass
class RetrievedChunk:
    """Один результат поиска в ChromaDB.

    source_id присваивается на один вызов retrieval (S1, S2, ...),
    а не хранится постоянно вместе с чанком.
    """

    chunk_id: str
    text: str
    source_name: str
    source_type: str
    page: int | None
    section: str | None
    score: float
    source_id: str


@dataclass
class SourceRef:
    """Нормализованная ссылка на источник для отображения в UI."""

    source_id: str
    source_name: str
    locator: str  # "стр. 12" | "раздел «...»" | "строки 10-20" | "фрагмент N"
    snippet: str
    chunk_id: str
    score: float | None = None


@dataclass
class AnswerResult:
    """Единый результат, который сервисный слой отдаёт интерфейсу."""

    status: str
    answer_text: str
    sources: list[SourceRef] = field(default_factory=list)
    reason: str | None = None
    retrieval_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Недопустимый статус AnswerResult: {self.status!r}")


@dataclass
class IndexStats:
    """Результат пакетной индексации для отображения в UI."""

    files_total: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
