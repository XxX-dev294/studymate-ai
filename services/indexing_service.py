"""
services/indexing_service.py

Склеивает Loader -> Chunker -> EmbeddingModel -> VectorStore.
Контракт: IndexingService.index_files(files) -> IndexStats
"""

from __future__ import annotations

from core.schemas import IndexStats
from ingestion.chunker import Chunker
from ingestion.loader import DocumentLoader, UploadedFileLike
from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


class IndexingService:
    """Оркестрирует полный цикл индексации пакета файлов."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: Chunker | None = None,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or Chunker()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()

    def index_files(self, files: list[UploadedFileLike]) -> IndexStats:
        stats = IndexStats(files_total=len(files))
        if not files:
            return stats

        records, load_errors = self.loader.load(files)
        stats.errors.extend(load_errors)

        # Файлы, которые реально дали хотя бы один record, считаем
        # проиндексированными; остальные — пропущенными.
        successfully_loaded_names = {r.source_name for r in records}
        stats.files_indexed = len(successfully_loaded_names)
        stats.files_skipped = stats.files_total - stats.files_indexed

        if not records:
            return stats

        # Удалить старые чанки для документов, которые переиндексируются,
        # чтобы избежать дублей при обновлении файла (document_id меняется
        # при изменении содержимого файла, т.к. это sha256 содержимого;
        # поэтому фактическое удаление требуется только если пользователь
        # явно переиндексирует тот же контент — upsert по chunk_id уже
        # защищает от дублей в пределах одного content_hash).
        chunks = self.chunker.split(records)
        if not chunks:
            stats.errors.append(
                "Текст успешно извлечён, но не удалось создать ни одного чанка."
            )
            return stats

        texts = [c.text for c in chunks]
        try:
            embeddings = self.embedding_model.embed_passages(texts)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"Ошибка создания эмбеддингов: {exc}")
            stats.files_indexed = 0
            stats.files_skipped = stats.files_total
            return stats

        try:
            saved = self.vector_store.upsert(chunks, embeddings)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"Ошибка записи в ChromaDB: {exc}")
            stats.files_indexed = 0
            stats.files_skipped = stats.files_total
            return stats

        stats.chunks_created = saved
        return stats

    def get_index_overview(self) -> dict:
        """Статистика текущего индекса для статусной панели UI."""
        try:
            count = self.vector_store.count()
        except Exception:  # noqa: BLE001
            count = 0
        return {"chunks_total": count}
