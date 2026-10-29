"""
tests/test_chunker.py

Проверяет чанкинг и требование "PDF делится внутри каждой
страницы, не через границу страниц".
"""

from __future__ import annotations

import unittest

from core.schemas import DocumentRecord
from ingestion.chunker import Chunker


class TestChunker(unittest.TestCase):
    def test_short_text_is_single_chunk(self) -> None:
        chunker = Chunker(chunk_size=800, overlap=120)
        record = DocumentRecord(
            document_id="doc1",
            source_name="a.txt",
            source_type="txt",
            text="Короткий текст.",
            content_hash="hash1",
        )
        chunks = chunker.split([record])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].source_name, "a.txt")

    def test_long_text_is_split_with_overlap(self) -> None:
        chunker = Chunker(chunk_size=100, overlap=20)
        text = " ".join(f"слово{i}" for i in range(200))
        record = DocumentRecord(
            document_id="doc1",
            source_name="a.txt",
            source_type="txt",
            text=text,
            content_hash="hash1",
        )
        chunks = chunker.split([record])
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 100)
        # chunk_index идёт по порядку 0,1,2,...
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_chunks_do_not_cross_page_boundary(self) -> None:
        chunker = Chunker(chunk_size=50, overlap=10)
        page1 = DocumentRecord(
            document_id="doc1",
            source_name="lecture.pdf",
            source_type="pdf",
            text="A" * 80,
            page=1,
            content_hash="hash1",
        )
        page2 = DocumentRecord(
            document_id="doc1",
            source_name="lecture.pdf",
            source_type="pdf",
            text="B" * 80,
            page=2,
            content_hash="hash1",
        )
        chunks = chunker.split([page1, page2])
        for c in chunks:
            # Ни один чанк не должен содержать символы и страницы 1, и страницы 2.
            self.assertTrue(set(c.text) <= {"A"} or set(c.text) <= {"B"})
        pages_seen = {c.page for c in chunks}
        self.assertEqual(pages_seen, {1, 2})

    def test_chunk_metadata_matches_source_record(self) -> None:
        chunker = Chunker(chunk_size=800, overlap=120)
        record = DocumentRecord(
            document_id="doc42",
            source_name="notes.md",
            source_type="md",
            text="Текст секции.",
            section="Введение",
            content_hash="hashXYZ",
        )
        chunks = chunker.split([record])
        c = chunks[0]
        self.assertEqual(c.document_id, "doc42")
        self.assertEqual(c.section, "Введение")
        self.assertEqual(c.content_hash, "hashXYZ")
        self.assertTrue(c.chunk_id)

    def test_same_content_produces_same_chunk_id(self) -> None:
        """Повторный импорт неизменённого файла не должен создавать дубли."""
        chunker = Chunker(chunk_size=800, overlap=120)
        record = DocumentRecord(
            document_id="doc1",
            source_name="a.txt",
            source_type="txt",
            text="Стабильный текст.",
            content_hash="samehash",
        )
        chunks1 = chunker.split([record])
        chunks2 = chunker.split([record])
        self.assertEqual(chunks1[0].chunk_id, chunks2[0].chunk_id)

    def test_empty_text_produces_no_chunks(self) -> None:
        chunker = Chunker()
        record = DocumentRecord(
            document_id="doc1",
            source_name="empty.txt",
            source_type="txt",
            text="   ",
            content_hash="h",
        )
        self.assertEqual(chunker.split([record]), [])

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            Chunker(chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
