"""
retrieval/embeddings.py

Обёртка над sentence-transformers. Контракт:
EmbeddingModel.embed_passages(texts) -> list[list[float]]
EmbeddingModel.embed_query(question) -> list[float]

Используются префиксы "query: " / "passage: ", как требует карточка
модели intfloat/multilingual-e5-small.
"""

from __future__ import annotations

from config import settings

_MODEL_CACHE: dict[str, object] = {}


class EmbeddingModel:
    """Локальная модель эмбеддингов (multilingual-e5-small по умолчанию)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self._model = None  # ленивая загрузка — не блокировать импорт модуля

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Библиотека sentence-transformers не установлена. "
                "Выполните: pip install sentence-transformers"
            ) from exc
        model = SentenceTransformer(self.model_name)
        _MODEL_CACHE[self.model_name] = model
        self._model = model
        return model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Эмбеддинги для текстов чанков (индексация)."""
        if not texts:
            return []
        model = self._load()
        prefixed = [f"{settings.embedding_passage_prefix}{t}" for t in texts]
        vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, question: str) -> list[float]:
        """Эмбеддинг для вопроса пользователя (поиск)."""
        model = self._load()
        prefixed = f"{settings.embedding_query_prefix}{question}"
        vector = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
        return vector.tolist()

    @property
    def dimension(self) -> int | None:
        if self._model is None:
            return None
        return self._model.get_sentence_embedding_dimension()
