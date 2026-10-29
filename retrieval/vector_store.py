"""
retrieval/vector_store.py

CRUD поверх ChromaDB PersistentClient. Контракт:
VectorStore.upsert(chunks, embeddings) -> int (кол-во сохранённых чанков)
VectorStore.query(query_embedding, top_k, filters=None) -> raw chroma result
VectorStore.count() -> int
VectorStore.delete_by_document_id(document_id) -> None

В metadata Chroma
хранятся только простые типы; None-поля заменяются на "" / -1, т.к.
не все версии Chroma принимают None как значение метаданных.
"""

from __future__ import annotations

from config import settings
from core.schemas import DocumentChunk

_NO_PAGE = -1
_EMPTY_SECTION = ""


def _page_to_meta(page: int | None) -> int:
    return page if page is not None else _NO_PAGE


def _page_from_meta(value: int | None) -> int | None:
    if value is None or value == _NO_PAGE:
        return None
    return int(value)


def _section_to_meta(section: str | None) -> str:
    return section if section else _EMPTY_SECTION


def _section_from_meta(value: str | None) -> str | None:
    if not value:
        return None
    return value


class VectorStore:
    """Локальное персистентное хранилище векторов на базе ChromaDB."""

    def __init__(
        self, path: str | None = None, collection_name: str | None = None
    ) -> None:
        self.path = path or settings.chroma_path
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Библиотека chromadb не установлена. Выполните: pip install chromadb"
            ) from exc

        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": settings.embedding_model_name},
        )
        return self._collection

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Количество чанков и эмбеддингов должно совпадать.")
        if not chunks:
            return 0

        collection = self._get_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "source_name": c.source_name,
                "source_type": c.source_type,
                "page": _page_to_meta(c.page),
                "section": _section_to_meta(c.section),
                "chunk_index": c.chunk_index,
                "content_hash": c.content_hash,
                "indexed_at": c.indexed_at,
                "embedding_model": settings.embedding_model_name,
            }
            for c in chunks
        ]
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        return len(chunks)

    def delete_by_document_id(self, document_id: str) -> None:
        collection = self._get_collection()
        collection.delete(where={"document_id": document_id})

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ):
        collection = self._get_collection()
        if collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(collection.count(), 1)),
            where=filters,
        )

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def existing_chunk_ids(self, document_id: str) -> set[str]:
        collection = self._get_collection()
        result = collection.get(where={"document_id": document_id})
        return set(result.get("ids", []))

    def get_chunks_by_document_id(self, document_id: str) -> dict:
        """Возвращает все чанки документа (для режима конспекта по файлу)."""
        collection = self._get_collection()
        return collection.get(where={"document_id": document_id})

    def list_documents(self) -> list[dict]:
        """Список уникальных проиндексированных документов (для UI выбора файла)."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in result.get("metadatas", []):
            if not meta:
                continue
            doc_id = meta.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "source_name": meta.get("source_name", "unknown"),
                    "source_type": meta.get("source_type", "unknown"),
                }
        return list(seen.values())

    @staticmethod
    def page_from_meta(value) -> int | None:
        return _page_from_meta(value)

    @staticmethod
    def section_from_meta(value) -> str | None:
        return _section_from_meta(value)
