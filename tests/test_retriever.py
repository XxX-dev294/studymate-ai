"""
tests/test_retriever.py

Проверяет similarity gate и поведение при пустой коллекции, используя
фейковую векторную БД (без реального ChromaDB/sentence-transformers).
"""

from __future__ import annotations

import unittest

from retrieval.retriever import Retriever
from tests.fixtures import FakeEmbeddingModel


class FakeVectorStore:
    """Имитирует VectorStore.query()/count() с заранее заданными distances."""

    def __init__(self, chunk_specs: list[dict], total_count: int | None = None):
        self._chunk_specs = chunk_specs
        self._total_count = total_count if total_count is not None else len(chunk_specs)

    def count(self) -> int:
        return self._total_count

    def query(self, query_embedding, top_k, filters=None):
        specs = self._chunk_specs[:top_k]
        return {
            "ids": [[s["id"] for s in specs]],
            "documents": [[s["text"] for s in specs]],
            "metadatas": [
                [
                    {
                        "source_name": s.get("source_name", "doc.pdf"),
                        "source_type": "pdf",
                        "page": s.get("page", 1),
                        "section": "",
                    }
                    for s in specs
                ]
            ],
            "distances": [[s["distance"] for s in specs]],
        }


class TestRetriever(unittest.TestCase):
    def test_empty_collection_returns_empty_list(self) -> None:
        store = FakeVectorStore(chunk_specs=[], total_count=0)
        retriever = Retriever(embedding_model=FakeEmbeddingModel(), vector_store=store)
        results = retriever.search("любой вопрос")
        self.assertEqual(results, [])

    def test_empty_question_returns_empty_list(self) -> None:
        store = FakeVectorStore(chunk_specs=[{"id": "c1", "text": "t", "distance": 0.1}])
        retriever = Retriever(embedding_model=FakeEmbeddingModel(), vector_store=store)
        self.assertEqual(retriever.search("   "), [])

    def test_results_below_threshold_are_filtered_out(self) -> None:
        # distance 0.1 -> similarity 0.9 (проходит), distance 0.9 -> similarity 0.1 (не проходит)
        store = FakeVectorStore(
            chunk_specs=[
                {"id": "c1", "text": "релевантный фрагмент", "distance": 0.1},
                {"id": "c2", "text": "нерелевантный фрагмент", "distance": 0.9},
            ]
        )
        retriever = Retriever(embedding_model=FakeEmbeddingModel(), vector_store=store)
        results = retriever.search("вопрос", min_similarity=0.6)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")

    def test_results_sorted_by_score_descending_with_sequential_source_ids(self) -> None:
        store = FakeVectorStore(
            chunk_specs=[
                {"id": "c1", "text": "a", "distance": 0.3},  # similarity 0.7
                {"id": "c2", "text": "b", "distance": 0.1},  # similarity 0.9
            ]
        )
        retriever = Retriever(embedding_model=FakeEmbeddingModel(), vector_store=store)
        results = retriever.search("вопрос", min_similarity=0.0)
        self.assertEqual(results[0].chunk_id, "c2")  # более релевантный первым
        self.assertEqual(results[0].source_id, "S1")
        self.assertEqual(results[1].source_id, "S2")

    def test_page_metadata_preserved(self) -> None:
        store = FakeVectorStore(
            chunk_specs=[{"id": "c1", "text": "t", "distance": 0.0, "page": 12}]
        )
        retriever = Retriever(embedding_model=FakeEmbeddingModel(), vector_store=store)
        results = retriever.search("вопрос", min_similarity=0.0)
        self.assertEqual(results[0].page, 12)


if __name__ == "__main__":
    unittest.main()
