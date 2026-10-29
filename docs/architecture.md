# Архитектура StudyMate AI

## Поток данных

### Индексация (offline)

```
Streamlit uploader
  -> UploadedFileLike (ui/sidebar.py)
  -> IndexingService.index_files() (services/indexing_service.py)
       -> DocumentLoader.load()        (ingestion/loader.py)
          -> list[DocumentRecord]       (постранично для PDF, по секциям для MD)
       -> Chunker.split()               (ingestion/chunker.py)
          -> list[DocumentChunk]        (800 симв., overlap 120, не через границу страницы)
       -> EmbeddingModel.embed_passages() (retrieval/embeddings.py)
          -> list[list[float]]
       -> VectorStore.upsert()           (retrieval/vector_store.py)
          -> ChromaDB persistent collection
  -> IndexStats -> UI
```

### Вопрос пользователя (online)

```
Streamlit chat_input
  -> AnswerService.answer() (services/answer_service.py)
       -> Retriever.search()            (retrieval/retriever.py)
          -> EmbeddingModel.embed_query() + VectorStore.query()
          -> similarity gate (MIN_SIMILARITY)
          -> list[RetrievedChunk] с S1..Sn
       -> [если пусто] -> фиксированный отказ (status=no_context), LLM не вызывается
       -> PromptBuilder (llm/prompts.py) -> OllamaClient.generate() (llm/ollama_client.py)
       -> валидация [Sx] относительно valid_ids
          -> [если невалидно] одна повторная генерация
          -> [если снова невалидно] status=invalid_citations, безопасный текст
       -> SourceRef[] строится сервером из метаданных RetrievedChunk, не из текста LLM
       -> AnswerResult -> UI (ответ + карточки источников)
```

### Конспект

```
UI (режим "По теме" | "По файлу")
  -> SummaryService.summarize() (services/summary_service.py)
       -> "По теме": Retriever.search(topic, top_k=6)
       -> "По файлу": VectorStore.get_chunks_by_document_id() + fit_to_context()
       -> PromptBuilder (summary) -> OllamaClient.generate()
       -> источники из метаданных -> AnswerResult -> UI
```

## Границы модулей

- UI (`ui/`) не вызывает `retrieval/` или `llm/` напрямую — только через
  `services/`.
- `ingestion/` и `retrieval/` не меняют формат `AnswerResult`/
  `RetrievedChunk` без согласования.
- `llm/` и `services/answer_service.py` не обращаются к облачным API —
  только к локальному Ollama (`config.settings.ollama_host`).
- Источники (`SourceRef`) всегда строятся из метаданных retrieval на
  сервере, никогда не парсятся из свободного текста, сгенерированного
  LLM — это ключевая защита от ложных цитат (см. `services/answer_service.py`,
  раздел "Валидация цитат").

## Статусы `AnswerResult.status`

| Статус | Когда возникает | LLM вызывается? |
|---|---|---|
| `ok` | Найден релевантный контекст, ответ сгенерирован и прошёл валидацию ссылок | да |
| `no_context` | Retrieval не нашёл ничего выше порога `MIN_SIMILARITY` | нет |
| `index_empty` | В коллекции ChromaDB 0 чанков | нет |
| `llm_unavailable` | Ollama недоступна (сеть/сервис/модель) | попытка была |
| `invalid_citations` | Модель дважды подряд сослалась на несуществующий `[Sx]` | да (дважды) |
| `error` | Пустой вопрос или иная ошибка валидации входа | нет |

## Не реализовано

MCP-расширение (инструменты `list_indexed_documents`, `search_materials`,
`get_source_excerpt`, `save_study_note`) пока только в планах, в коде
соответствующего сервера нет.
