"""
tests/test_citations.py

Проверяет политику цитирования: извлечение [Sx] из текста LLM,
построение SourceRef только из метаданных retrieval.
"""

from __future__ import annotations

import unittest

from services.answer_service import _build_source_refs, _extract_citation_ids
from tests.fixtures import make_retrieved_chunk


class TestCitationExtraction(unittest.TestCase):
    def test_extracts_single_citation(self) -> None:
        self.assertEqual(_extract_citation_ids("Ответ со ссылкой [S1]."), {"S1"})

    def test_extracts_multiple_unique_citations(self) -> None:
        text = "Факт один [S1]. Факт два [S2]. Повтор [S1]."
        self.assertEqual(_extract_citation_ids(text), {"S1", "S2"})

    def test_no_citations_returns_empty_set(self) -> None:
        self.assertEqual(_extract_citation_ids("Ответ без ссылок."), set())

    def test_detects_invalid_citation_not_in_valid_set(self) -> None:
        text = "Ссылка на несуществующий [S99] источник."
        used = _extract_citation_ids(text)
        valid = {"S1", "S2"}
        self.assertEqual(used - valid, {"S99"})


class TestSourceRefBuilding(unittest.TestCase):
    def test_source_ref_uses_page_locator_for_pdf(self) -> None:
        chunk = make_retrieved_chunk(source_id="S1", page=12, source_name="Lecture_01.pdf")
        refs = _build_source_refs([chunk], limit=4)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].locator, "стр. 12")
        self.assertEqual(refs[0].source_name, "Lecture_01.pdf")

    def test_source_ref_respects_limit(self) -> None:
        chunks = [make_retrieved_chunk(source_id=f"S{i}") for i in range(1, 10)]
        refs = _build_source_refs(chunks, limit=4)
        self.assertEqual(len(refs), 4)

    def test_snippet_is_truncated_for_long_text(self) -> None:
        long_text = "x" * 1000
        chunk = make_retrieved_chunk(text=long_text)
        refs = _build_source_refs([chunk], limit=4)
        self.assertLessEqual(len(refs[0].snippet), 282)  # 280 + многоточие


if __name__ == "__main__":
    unittest.main()
