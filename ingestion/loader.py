"""
ingestion/loader.py

Загрузка PDF/TXT/MD файлов в list[DocumentRecord].
Контракт: DocumentLoader.load(files) -> (list[DocumentRecord], list[str] errors)

Требования к загрузке:
- разрешены .pdf, .txt, .md без учёта регистра расширения.
- проверка размера, пустого содержимого, дубликатов, читаемости.
- PDF — постраничное извлечение через pypdf, номер страницы с 1,
  метаданные назначаются ДО чанкинга (здесь, на уровне DocumentRecord).
- TXT/MD — сначала UTF-8/UTF-8-SIG, при ошибке — CP1251 с
  предупреждением; для Markdown сохранять заголовки "#" как section.
- ошибка одного файла не должна прерывать обработку остальных.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from config import settings
from core.schemas import (
    SOURCE_TYPE_MD,
    SOURCE_TYPE_PDF,
    SOURCE_TYPE_TXT,
    DocumentRecord,
)

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover - проверяется наличием зависимости
    PdfReader = None
    PdfReadError = Exception


@dataclass
class UploadedFileLike:
    """Минимальный интерфейс загруженного файла.

    Совместим как со Streamlit UploadedFile (есть .name и .getvalue()/.read()),
    так и с обычными файлами на диске, открытыми в бинарном режиме, через
    from_path().
    """

    name: str
    data: bytes

    @classmethod
    def from_streamlit(cls, uploaded_file) -> "UploadedFileLike":
        return cls(name=uploaded_file.name, data=uploaded_file.getvalue())

    @classmethod
    def from_path(cls, path: str) -> "UploadedFileLike":
        import os

        with open(path, "rb") as fh:
            data = fh.read()
        return cls(name=os.path.basename(path), data=data)


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _ext_of(name: str) -> str:
    idx = name.rfind(".")
    if idx == -1:
        return ""
    return name[idx:].lower()


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _decode_text(data: bytes) -> tuple[str, str | None]:
    """Декодирует байты в текст. Возвращает (текст, предупреждение|None).

    Порядок по - сначала UTF-8 / UTF-8-SIG, при ошибке — CP1251.
    """
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc), None
        except UnicodeDecodeError:
            continue
    try:
        return data.decode("cp1251"), (
            "Кодировка файла не UTF-8; использована CP1251 как запасной вариант."
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"Не удалось определить кодировку файла: {exc}") from exc


class DocumentLoader:
    """Загружает PDF/TXT/MD файлы и превращает их в DocumentRecord."""

    def __init__(self, max_file_mb: int | None = None) -> None:
        self.max_file_mb = max_file_mb if max_file_mb is not None else settings.upload_max_mb

    def load(
        self, files: list[UploadedFileLike]
    ) -> tuple[list[DocumentRecord], list[str]]:
        """Загружает пакет файлов.

        Возвращает (records, errors). Ошибка одного файла не прерывает
        обработку остальных.
        """
        records: list[DocumentRecord] = []
        errors: list[str] = []
        seen_hashes: dict[str, str] = {}

        for f in files:
            try:
                file_records, warning = self._load_one(f, seen_hashes)
                records.extend(file_records)
                if warning:
                    errors.append(f"{f.name}: {warning} (файл всё же обработан)")
            except Exception as exc:  # noqa: BLE001 — изолируем ошибку одного файла
                errors.append(f"{f.name}: {exc}")

        return records, errors

    def _load_one(
        self, f: UploadedFileLike, seen_hashes: dict[str, str]
    ) -> tuple[list[DocumentRecord], str | None]:
        ext = _ext_of(f.name)
        if ext not in settings.allowed_extensions:
            raise ValueError(
                f"Неподдерживаемый тип файла '{ext}'. Разрешены: "
                f"{', '.join(settings.allowed_extensions)}."
            )

        size_mb = len(f.data) / (1024 * 1024)
        if size_mb > self.max_file_mb:
            raise ValueError(
                f"Файл превышает лимит {self.max_file_mb} МБ "
                f"(размер: {size_mb:.1f} МБ)."
            )

        if len(f.data) == 0:
            raise ValueError("Файл пустой.")

        content_hash = _sha256(f.data)
        if content_hash in seen_hashes:
            raise ValueError(
                f"Дубликат уже загруженного файла '{seen_hashes[content_hash]}'."
            )
        seen_hashes[content_hash] = f.name

        document_id = content_hash

        if ext == ".pdf":
            return self._load_pdf(f, document_id, content_hash), None
        if ext == ".md":
            records, warning = self._load_markdown(f, document_id, content_hash)
            return records, warning
        if ext == ".txt":
            records, warning = self._load_txt(f, document_id, content_hash)
            return records, warning

        raise ValueError(f"Неподдерживаемое расширение '{ext}'.")  # защитный код

    def _load_pdf(
        self, f: UploadedFileLike, document_id: str, content_hash: str
    ) -> list[DocumentRecord]:
        if PdfReader is None:
            raise RuntimeError(
                "Библиотека pypdf не установлена. Выполните: pip install pypdf"
            )

        try:
            reader = PdfReader(io.BytesIO(f.data))
        except PdfReadError as exc:
            raise ValueError(f"Повреждённый или нечитаемый PDF: {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"PDF защищён паролем и не может быть прочитан: {exc}"
                ) from exc

        records: list[DocumentRecord] = []
        any_text = False
        for i, page in enumerate(reader.pages):
            # Номер страницы для пользователя — 1-based; прибавление единицы
            # происходит один раз, здесь, при создании метаданных.
            page_number = i + 1
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 — одна битая страница не должна всё ломать
                text = ""
            text = text.strip()
            if text:
                any_text = True
            records.append(
                DocumentRecord(
                    document_id=document_id,
                    source_name=f.name,
                    source_type=SOURCE_TYPE_PDF,
                    text=text,
                    page=page_number,
                    section=None,
                    content_hash=content_hash,
                )
            )

        if not any_text:
            raise ValueError(
                "Не удалось извлечь текст из PDF (возможно, это скан без OCR; "
                "OCR не входит в MVP)."
            )

        # Отбрасываем страницы без текста — пустой текст нечего чанковать,
        # но страницы с текстом сохраняют верную нумерацию, т.к. она
        # назначена до фильтрации.
        return [r for r in records if r.text]

    def _load_txt(
        self, f: UploadedFileLike, document_id: str, content_hash: str
    ) -> tuple[list[DocumentRecord], str | None]:
        text, warning = _decode_text(f.data)
        text = text.strip()
        if not text:
            raise ValueError("TXT-файл не содержит текста.")

        lines = text.splitlines()
        record = DocumentRecord(
            document_id=document_id,
            source_name=f.name,
            source_type=SOURCE_TYPE_TXT,
            text=text,
            page=None,
            section=None,
            line_start=1,
            line_end=len(lines),
            content_hash=content_hash,
        )
        return [record], warning

    def _load_markdown(
        self, f: UploadedFileLike, document_id: str, content_hash: str
    ) -> tuple[list[DocumentRecord], str | None]:
        text, warning = _decode_text(f.data)
        if not text.strip():
            raise ValueError("Markdown-файл не содержит текста.")

        records = self._split_markdown_by_headings(text, f.name, document_id, content_hash)
        return records, warning

    def _split_markdown_by_headings(
        self, text: str, source_name: str, document_id: str, content_hash: str
    ) -> list[DocumentRecord]:
        """Делит markdown на секции по заголовкам "#", сохраняя line_start/end."""
        lines = text.splitlines()
        sections: list[tuple[str | None, int, int, list[str]]] = []
        current_heading: str | None = None
        current_start = 1
        buffer: list[str] = []

        def flush(end_line: int) -> None:
            if buffer and any(line.strip() for line in buffer):
                sections.append((current_heading, current_start, end_line, list(buffer)))

        for idx, line in enumerate(lines, start=1):
            match = _MD_HEADING_RE.match(line)
            if match:
                flush(idx - 1)
                current_heading = match.group(2).strip()
                current_start = idx
                buffer = [line]
            else:
                buffer.append(line)
        flush(len(lines))

        if not sections:
            # Нет заголовков — весь файл одна секция.
            return [
                DocumentRecord(
                    document_id=document_id,
                    source_name=source_name,
                    source_type=SOURCE_TYPE_MD,
                    text=text.strip(),
                    page=None,
                    section=None,
                    line_start=1,
                    line_end=len(lines),
                    content_hash=content_hash,
                )
            ]

        records = []
        for heading, start, end, buf_lines in sections:
            section_text = "\n".join(buf_lines).strip()
            if not section_text:
                continue
            records.append(
                DocumentRecord(
                    document_id=document_id,
                    source_name=source_name,
                    source_type=SOURCE_TYPE_MD,
                    text=section_text,
                    page=None,
                    section=heading,
                    line_start=start,
                    line_end=end,
                    content_hash=content_hash,
                )
            )
        return records
