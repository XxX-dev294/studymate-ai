"""
retrieval/retriever.py

Контракт: Retriever.search(question, top_k, filters=None) -> list[RetrievedChunk]

- Косинусная метрика; Chroma с HNSW по умолчанию возвращает cosine distance,
  similarity = 1 - distance.
- При пустой коллекции возвращает пустой список, не выбрасывая исключение
  (проверку на index_empty делает сервисный слой через VectorStore.count()).
- Возвращённая структура одинакова независимо от UI и LLM.
"""

from __future__ import annotations

from config import settings
from core.schemas import RetrievedChunk
from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


def _distance_to_similarity(distance: float) -> float:
    """Cosine distance -> similarity в диапазоне примерно [0, 1]."""
    return 1.0 - distance


class Retriever:
    """Семантический поиск top-k по ChromaDB с порогом релевантности."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()

    def search(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            return []

        if self.vector_store.count() == 0:
            return []

        k = top_k or settings.top_k
        threshold = (
            min_similarity if min_similarity is not None else settings.min_similarity
        )

        query_embedding = self.embedding_model.embed_query(question)
        raw = self.vector_store.query(query_embedding, top_k=k, filters=filters)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            distance = distances[idx] if idx < len(distances) else 1.0
            similarity = _distance_to_similarity(distance)
            if similarity < threshold:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=documents[idx],
                    source_name=metadata.get("source_name", "unknown"),
                    source_type=metadata.get("source_type", "unknown"),
                    page=VectorStore.page_from_meta(metadata.get("page")),
                    section=VectorStore.section_from_meta(metadata.get("section")),
                    score=round(similarity, 4),
                    source_id=f"S{idx + 1}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        # Пересобрать source_id по отсортированному порядку S1..Sn.
        for i, r in enumerate(results):
            r.source_id = f"S{i + 1}"
        return results
