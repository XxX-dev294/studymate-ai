"""
config.py

Централизованная конфигурация StudyMate AI. Все параметры чанкинга,
моделей, top-k, порога, температуры и путей задаются здесь и/или через
переменные окружения — модули не должны хардкодить эти значения у себя.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Приложение ---
    app_name: str = "StudyMate AI"

    # --- Чанкинг ---
    chunk_size: int = _env_int("STUDYMATE_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("STUDYMATE_CHUNK_OVERLAP", 120)

    # --- Retrieval ---
    top_k: int = _env_int("STUDYMATE_TOP_K", 4)
    min_similarity: float = _env_float("STUDYMATE_MIN_SIMILARITY", 0.60)

    # --- LLM / Ollama ---
    ollama_host: str = _env_str("STUDYMATE_OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = _env_str("STUDYMATE_LLM_MODEL", "qwen3:4b")
    llm_temperature: float = _env_float("STUDYMATE_LLM_TEMPERATURE", 0.2)
    llm_summary_temperature: float = _env_float("STUDYMATE_LLM_SUMMARY_TEMPERATURE", 0.3)
    llm_timeout_seconds: int = _env_int("STUDYMATE_LLM_TIMEOUT", 120)
    max_context_chars: int = _env_int("STUDYMATE_MAX_CONTEXT_CHARS", 12_000)

    # --- Embeddings ---
    embedding_model_name: str = _env_str(
        "STUDYMATE_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
    )
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # --- Загрузка файлов ---
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".md")
    upload_max_mb: int = _env_int("STUDYMATE_UPLOAD_MAX_MB", 50)

    # --- Хранилище ---
    chroma_path: str = _env_str("STUDYMATE_CHROMA_PATH", "./data/chroma")
    chroma_collection_name: str = _env_str(
        "STUDYMATE_CHROMA_COLLECTION", "studymate_chunks"
    )
    notes_dir: str = _env_str("STUDYMATE_NOTES_DIR", "./data/notes")

    # --- Источники / цитирование ---
    max_sources_in_answer: int = 4

    # --- Конспект ---
    summary_max_points: int = 7


settings = Settings()
