# StudyMate AI

Локальный помощник для поиска, объяснения и конспектирования учебных
материалов. Приложение индексирует PDF/TXT/MD файлы в локальной векторной
базе (ChromaDB), ищет релевантные фрагменты и отвечает на вопросы через
локальную LLM (Ollama) со ссылками на файл и страницу/раздел. Все
вычисления выполняются на локальной машине, без обращения к облачным API.

---

## 1. Возможности

- Загрузка PDF / TXT / MD (`ingestion/loader.py`) с проверкой размера,
  пустого содержимого, дубликатов (SHA-256) и кодировки (UTF-8 →
  CP1251 fallback). Страницы PDF нумеруются с 1, Markdown делится по
  заголовкам `#`.
- Чанкинг с метаданными (`ingestion/chunker.py`): по умолчанию 800
  символов с перекрытием 120, чанки не пересекают границу страницы PDF,
  `chunk_id` стабильный, поэтому повторный импорт не создаёт дублей.
- Контракты данных (`core/schemas.py`): `DocumentRecord`, `DocumentChunk`,
  `RetrievedChunk`, `SourceRef`, `AnswerResult`, `IndexStats`.
- Эмбеддинги (`retrieval/embeddings.py`): `sentence-transformers` и
  `intfloat/multilingual-e5-small` с префиксами `query:` / `passage:`.
- Векторное хранилище (`retrieval/vector_store.py`): ChromaDB
  `PersistentClient`, upsert/query/delete по `document_id`.
- Семантический поиск с порогом релевантности (`retrieval/retriever.py`).
- RAG-пайплайн с цитированием и защитой от галлюцинаций
  (`services/answer_service.py`): фиксированный отказ при отсутствии
  контекста, проверка `[Sx]`-ссылок относительно реально найденных
  чанков, одна повторная попытка при невалидных ссылках, затем безопасный
  отказ.
- Режим конспекта по теме или по файлу (`services/summary_service.py`),
  с разбиением на части, если контекст не помещается в лимит.
- Клиент Ollama (`llm/ollama_client.py`) и промпты (`llm/prompts.py`),
  которые трактуют содержимое документов как данные, а не инструкции
  (защита от prompt injection).
- Streamlit-интерфейс (`app.py`, `ui/`): боковая панель с загрузкой и
  статусом, вкладки «Чат» и «Конспект», карточки источников с
  раскрываемым фрагментом.
- Юнит-тесты (`tests/`): загрузчик, чанкер, retriever, проверка цитат,
  защита от невалидных ответов.

### Ограничения

- Поиск глобальный, по всей коллекции; фильтра по типу или названию
  документа пока нет.
- Конспект есть только в режиме «Кратко».
- Нельзя скопировать результат конспекта кнопкой.
- Нет OCR для сканов PDF, GraphRAG/reranker, многопользовательского
  режима и MCP-расширения.
- `MIN_SIMILARITY = 0.60` — стартовое значение, его стоит подбирать под
  конкретный набор материалов.
- Модель по умолчанию `qwen3:4b` выбрана как отправная точка; `qwen3:8b`
  и `gemma3:4b` можно подставить через переменную окружения.
- Приложение рассчитано на одного локального пользователя и 10–30
  файлов.
- Защита от галлюцинаций снижает риск, но не исключает его полностью:
  работают порог релевантности, контекстный промпт и серверная проверка
  источников.

---

## 2. Структура репозитория

```
studymate-ai/
├── app.py                     # точка входа Streamlit
├── config.py                  # единая конфигурация
├── requirements.txt
├── core/
│   └── schemas.py             # общие типы данных
├── ingestion/
│   ├── loader.py              # PDF/TXT/MD -> DocumentRecord[]
│   └── chunker.py             # DocumentRecord[] -> DocumentChunk[]
├── retrieval/
│   ├── embeddings.py          # обёртка над sentence-transformers
│   ├── vector_store.py        # ChromaDB CRUD
│   └── retriever.py           # top-k + порог релевантности
├── llm/
│   ├── ollama_client.py       # HTTP-клиент Ollama
│   └── prompts.py             # системные промпты ответа/конспекта
├── services/
│   ├── indexing_service.py    # Loader -> Chunker -> Embeddings -> Store
│   ├── answer_service.py      # RAG-пайплайн + защита от галлюцинаций
│   └── summary_service.py     # режим конспекта
├── ui/
│   ├── sidebar.py             # загрузка, индексация, статус
│   └── components/
│       └── source_card.py     # карточки источников
├── tests/
│   ├── fixtures.py            # фейковые embeddings/Ollama, генератор PDF
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_citations.py
│   └── test_answer_guard.py
├── testdata/                  # синтетические учебные материалы
│   ├── sample_lecture.md
│   ├── sample_lecture.pdf
│   └── sample_notes.txt
├── docs/
│   └── architecture.md
└── data/                      # данные приложения, в .gitignore
    ├── chroma/
    └── notes/
```

---

## 3. Требования к окружению

- Python 3.11 (минимум — 3.10).
- [Ollama](https://ollama.com), установленный локально.
- Около 8 ГБ RAM для `qwen3:4b`; для `qwen3:8b` лучше 16 ГБ.
- Интернет нужен только один раз: для установки пакетов pip и первой
  загрузки моделей (Ollama и sentence-transformers). После этого
  приложение работает офлайн.

---

## 4. Установка

```bash
# 1. Перейти в папку проекта
cd studymate-ai

# 2. Создать и активировать виртуальное окружение
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Установить и запустить Ollama (см. https://ollama.com)
ollama serve &

# 5. Скачать модель
ollama pull qwen3:4b
# ollama pull qwen3:8b            # если хватает RAM
# ollama pull gemma3:4b           # запасной вариант

# 6. Один раз скачать модель эмбеддингов
python3 -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"
```

---

## 5. Запуск

```bash
source .venv/bin/activate
streamlit run app.py
```

Приложение откроется в браузере на `http://localhost:8501`. В боковой
панели показывается, доступна ли Ollama; если нет, интерфейс подскажет
команды (`ollama serve`, `ollama pull <модель>`).

После того как модели скачаны, сеть можно отключить: индексация, поиск,
генерация ответа и конспект работают без внешних API.

---

## 6. Тестирование

```bash
source .venv/bin/activate
pytest tests/ -v

# то же без pytest
python3 -m unittest discover -s tests -v
```

Тесты не требуют запущенного Ollama и не скачивают модели эмбеддингов:
вместо них используются лёгкие фейковые реализации из `tests/fixtures.py`,
поэтому тесты быстрые и воспроизводимые.

---

## 7. Команда и зоны ответственности

| Участник | Роль | Файлы |
|---|---|---|
| Участник 1 | Тимлид/архитектор: архитектура, интерфейсы модулей, интеграция, README | `app.py`, `config.py`, `core/schemas.py`, `services/` (интеграционный слой) |
| Участник 2 | AI/RAG-инженер: загрузка, чанкинг, эмбеддинги, ChromaDB, retrieval | `ingestion/`, `retrieval/` |
| Участник 3 | LLM & Prompt Engineer: Ollama-клиент, промпты, ответ/конспект, защита от галлюцинаций | `llm/`, `services/answer_service.py`, `services/summary_service.py` |
| Участник 4 | Frontend: Streamlit UX/UI | `ui/` |
| Участник 5 | QA/ресерчер: тесты, тестовый корпус, документация | `tests/`, `testdata/`, `docs/` |

---

## 8. Конфигурация

Все параметры лежат в `config.py` и переопределяются переменными
окружения с префиксом `STUDYMATE_` (`STUDYMATE_CHUNK_SIZE`,
`STUDYMATE_TOP_K`, `STUDYMATE_MIN_SIMILARITY`, `STUDYMATE_LLM_MODEL`,
`STUDYMATE_OLLAMA_HOST` и т.д.).

Пример смены модели:

```bash
STUDYMATE_LLM_MODEL=gemma3:4b streamlit run app.py
```
