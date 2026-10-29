"""
ingestion/chunker.py

Разбивает list[DocumentRecord] на list[DocumentChunk].
Контракт: Chunker.split(records) -> list[DocumentChunk]

По умолчанию 800 символов, overlap 120; PDF делится внутри каждой
страницы, не через границу страниц (каждый DocumentRecord уже
соответствует одной странице/секции, поэтому чанкер просто не
объединяет текст разных records).
Для каждого чанка сохраняются chunk_id, document_id,
source_name, source_type, page/section, chunk_index, text, content_hash.
"""

from __future__ import annotations

import hashlib

from config import settings
from core.schemas import DocumentChunk, DocumentRecord


def _stable_chunk_id(
    document_hash: str, locator: str, chunk_index: int, chunk_text: str
) -> str:
    """Стабильный ID: хеш документа + страница/раздел + индекс чанка + хеш текста.

    Благодаря этому повторный импорт неизменённого файла не создаёт дублей
    (ID получится тем же самым), а изменение текста изменит ID.
    """
    text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
    raw = f"{document_hash}|{locator}|{chunk_index}|{text_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Chunker:
    """Разбивает текст документов на фрагменты фиксированного размера."""

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap if overlap is not None else settings.chunk_overlap
        if self.overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")

    def split(self, records: list[DocumentRecord]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for record in records:
            chunks.extend(self._split_record(record))
        return chunks

    def _split_record(self, record: DocumentRecord) -> list[DocumentChunk]:
        text = record.text.strip()
        if not text:
            return []

        pieces = self._split_text(text)
        locator = f"page={record.page}|section={record.section}"

        out: list[DocumentChunk] = []
        for i, piece in enumerate(pieces):
            chunk_id = _stable_chunk_id(record.content_hash, locator, i, piece)
            out.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=record.document_id,
                    source_name=record.source_name,
                    source_type=record.source_type,
                    text=piece,
                    page=record.page,
                    section=record.section,
                    chunk_index=i,
                    content_hash=record.content_hash,
                )
            )
        return out

    def _split_text(self, text: str) -> list[str]:
        """Скользящее окно по символам с перекрытием, без разрыва слов."""
        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        start = 0
        n = len(text)
        step = self.chunk_size - self.overlap

        while start < n:
            end = min(start + self.chunk_size, n)
            # Не разрывать слово посередине: сдвинуть конец к ближайшему
            # пробелу слева, если это не конец текста.
            if end < n:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= n:
                break
            start += step
            if start <= 0:
                start = end  # защита от зацикливания при экстремальных настройках

        return pieces
