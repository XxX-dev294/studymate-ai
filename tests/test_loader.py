"""
tests/test_loader.py

Проверяет загрузку PDF/TXT/MD: расширения, кодировки, ошибки отдельных файлов.
"""

from __future__ import annotations

import unittest

from ingestion.loader import DocumentLoader, UploadedFileLike
from tests.fixtures import make_pdf_bytes


class TestDocumentLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DocumentLoader()

    # --- TXT ---
    def test_txt_utf8_loaded_with_line_range(self) -> None:
        content = "Строка 1\nСтрока 2\nСтрока 3".encode("utf-8")
        f = UploadedFileLike(name="notes.txt", data=content)
        records, errors = self.loader.load([f])
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r.source_type, "txt")
        self.assertEqual(r.line_start, 1)
        self.assertEqual(r.line_end, 3)
        self.assertIsNone(r.page)

    def test_txt_cp1251_fallback_produces_warning(self) -> None:
        content = "Привет мир".encode("cp1251")
        f = UploadedFileLike(name="legacy.txt", data=content)
        records, errors = self.loader.load([f])
        self.assertEqual(len(records), 1)
        self.assertTrue(any("CP1251" in e or "legacy.txt" in e for e in errors))

    def test_empty_txt_is_rejected_without_stopping_batch(self) -> None:
        good = UploadedFileLike(name="good.txt", data=b"Content")
        empty = UploadedFileLike(name="empty.txt", data=b"")
        records, errors = self.loader.load([good, empty])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_name, "good.txt")
        self.assertTrue(any("empty.txt" in e for e in errors))

    # --- Markdown ---
    def test_markdown_split_by_headings(self) -> None:
        content = (
            "# Введение\nТекст введения.\n\n"
            "## Раздел 1\nТекст раздела один.\n\n"
            "## Раздел 2\nТекст раздела два.\n"
        ).encode("utf-8")
        f = UploadedFileLike(name="doc.md", data=content)
        records, errors = self.loader.load([f])
        self.assertEqual(errors, [])
        sections = {r.section for r in records}
        self.assertIn("Введение", sections)
        self.assertIn("Раздел 1", sections)
        self.assertIn("Раздел 2", sections)
        for r in records:
            self.assertIsNotNone(r.line_start)
            self.assertIsNotNone(r.line_end)

    def test_markdown_without_headings_is_single_record(self) -> None:
        content = b"Just a plain paragraph without any heading."
        f = UploadedFileLike(name="plain.md", data=content)
        records, _ = self.loader.load([f])
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].section)

    # --- PDF ---
    def test_pdf_pages_get_correct_1_based_page_numbers(self) -> None:
        pdf_bytes = make_pdf_bytes(["Page one content", "Page two content", "Page three"])
        f = UploadedFileLike(name="lecture.pdf", data=pdf_bytes)
        records, errors = self.loader.load([f])
        self.assertEqual(errors, [])
        pages = sorted(r.page for r in records)
        self.assertEqual(pages, [1, 2, 3])
        for r in records:
            self.assertEqual(r.source_type, "pdf")
            self.assertTrue(r.text)

    # --- Общие проверки ---
    def test_unsupported_extension_is_rejected(self) -> None:
        f = UploadedFileLike(name="image.png", data=b"\x89PNG")
        records, errors = self.loader.load([f])
        self.assertEqual(records, [])
        self.assertTrue(any("image.png" in e for e in errors))

    def test_duplicate_file_detected_by_hash(self) -> None:
        data = b"Identical content for duplicate test."
        f1 = UploadedFileLike(name="a.txt", data=data)
        f2 = UploadedFileLike(name="b.txt", data=data)
        records, errors = self.loader.load([f1, f2])
        self.assertEqual(len(records), 1)
        self.assertTrue(any("Дубликат" in e for e in errors))

    def test_one_bad_file_does_not_block_others(self) -> None:
        bad = UploadedFileLike(name="bad.xyz", data=b"whatever")
        good = UploadedFileLike(name="good.txt", data=b"Valid content here.")
        records, errors = self.loader.load([bad, good])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_name, "good.txt")
        self.assertEqual(len(errors), 1)

    def test_oversized_file_is_rejected(self) -> None:
        small_loader = DocumentLoader(max_file_mb=0)  # лимит 0 МБ -> всё превышает
        f = UploadedFileLike(name="big.txt", data=b"some content")
        records, errors = small_loader.load([f])
        self.assertEqual(records, [])
        self.assertTrue(any("лимит" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
