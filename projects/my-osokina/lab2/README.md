# Лабораторная работа 2. RAG-агент по корпусу Arenadata Catalog

## Парсинг корпуса

`parse_to_markdown.py` читает только `.html`/`.pdf`, папки `*_files` (ассеты сохранённых страниц) пропускает. PDF идёт постранично через `pymupdf4llm`, в текст пишется маркер `<!-- Страница N -->`. HTML чистится BeautifulSoup и переводится в Markdown с ATX-заголовками. В начало файла — YAML: `source`, `path`, `date`, для HTML ещё `section`.

```bash
python parse_to_markdown.py
python parse_to_markdown.py --input ".../data_for_llm_2" --output ".../markdown_for_llm_2"
```

---

## Чанки и эмбеддинги

Скрипт: `scripts/build_index.py`.

### Сплиттеры

| `splitter` | Класс LangChain | Поведение |
|---|---|---|
| `recursive` | `RecursiveCharacterTextSplitter` | Страницы PDF, затем рекурсия по заголовкам/абзацам/предложениям с `overlap`. |
| `markdown` | `MarkdownHeaderTextSplitter` | Нарезка по **h1–h3**; слишком длинные секции дорезает `RecursiveCharacterTextSplitter` без overlap. |
| `semantic` | `SemanticChunker` (алиас `SemanticChunker`) | Семантические границы через dense-модель, затем рекурсия до `chunk_size`. |
| `hybrid` | MarkdownHeader + Recursive | Заголовки h1–h3 и окно с `overlap` внутри секции. |

Длина чанка считается **одним** `LengthCounter` (`src/length.py`): это max(число токенов) по токенайзерам всех моделей прогона (dense, cross-encoder). Лимит = min(`model_max_length`) − special tokens. 

`chunk_size` можно задать в конфиге, но фактически не больше лимита моделей.

Заголовки: `--headers-mode none|text|metadata|both`.

- `text` — путь заголовков в тексте чанка;
- `metadata` — только в payload;
- `both` — и туда, и туда;
- `none` — нигде.

### Эмбеддинги

`--vectorization dense|sparse|hybrid`

- **dense**: HuggingFace / Sentence-Transformers (`intfloat/multilingual-e5-small`, `e5-large-v2`, …). 
- **sparse**: `bm25`. 
- **hybrid**: dense + BM25, слияние через RRF.

```bash
pip install -r requirements.txt

python scripts/build_index.py
python scripts/build_index.py --splitter markdown --chunk-size 256 --overlap 40 --headers-mode text
python scripts/build_index.py --vectorization dense --dense-model intfloat/multilingual-e5-small --rebuild
```

Результат run:

- `chunks.jsonl` - сами чанки как структуры со следующими полями: id, text, source, path, date, section, page, headers, token_count
- `embeddings.npy` (если dense/hybrid)
- `bm25.pkl` (если sparse/hybrid и `sparse_model=bm25`)
- `config.yaml` снимка параметров

---

## Векторное хранилище

Скрипт: `scripts/load_to_vector_store.py`.

Схема точки: вектор + payload (`id`, `text`, `source`, `path`, `date`, `section`, `page`, `headers`, `token_count`, `run_id`). Имя коллекции: `rag_<run_id>` — эксперименты не перемешиваются.

Индекс HNSW, параметры:

- `--hnsw-m` - кол-во смежных вершин у каждой точки в HNSW-графе (16 (по умолчанию в Qdrant), 32, 48)
- `--hnsw-ef-construct` - кандидаты в смежные вершины (100–200)
- `--hnsw-ef-search`- кол-во лучших вершин во время обхода 64–256 (по умолчанию в Qdrant — 128)

Идемпотентность: upsert по стабильному UUID от `(collection, chunk_id)`. Повторная загрузка обновляет те же точки.

```bash
python scripts/load_to_vector_store.py
python scripts/load_to_vector_store.py --rebuild
python scripts/load_to_vector_store.py --drop-and-reindex --hnsw-m 16 --hnsw-ef-construct 64
```

`--drop-and-reindex` — reset коллекции текущего `run_id`.

---

## Метрики retrieval

Скрипт: `scripts/evaluate.py`. Набор вопросов: `eval/gold.json`.

Считаются Precision/Recall по чанкам и по документам плюс **MRR** (`--k`, по умолчанию `retrieval.top_k`). Ключи в логе: `precision_chunks@k`, `recall_chunks@k`, `precision_docs@k`, `recall_docs@k`, `mrr@k`.

- чанк релевантен, если `source` в gold и (если задано) текст содержит хотя бы одну из фраз;
- `recall_docs@k` — доля релевантных источников в top-k среди всех уникальных релевантных;
- `recall_chunks@k` — доля найденных релевантных чанков среди всех релевантных в корпусе.
- `precision_chunks@k` — доля релевантных чанков в top-k;
- `precision_docs@k` — доля релевантных источников в top-k среди уникальных источников в top-k;
- `mrr@k` — \(1/\text{ранг первого релевантного чанка}\).

```bash
python scripts/evaluate.py --k 5
python scripts/evaluate.py --k 10 --splitter hybrid --chunk-size 400
```

Строка с конфигом и метриками дописывается в `artifacts/experiments.jsonl`.

---

## RAG-движок

Скрипт: `scripts/ask.py`.

1. Векторизация вопроса (e5-префикс `query:` / BM25).
2. Поиск: dense в Qdrant, либо BM25; hybrid — Reciprocal Rank Fusion; опциональный cross-encoder.
4. Вызов локальной LLM через HTTP API Ollama (`/api/chat`).
5. Ответ модели + блок Цитаты (источник, страница если есть, сниппет).

```bash
ollama pull qwen2.5:3b
python scripts/ask.py "Что такое Data Lineage?"
python scripts/ask.py
```

Без запущенной Ollama скрипт всё равно вернёт цитаты из retrieval и сообщение об ошибке HTTP.

---

## Типичный прогон и подбор параметров

```bash
python parse_to_markdown.py
python scripts/build_index.py --splitter hybrid --chunk-size 400 --overlap 80 --headers-mode both --vectorization hybrid
python scripts/load_to_vector_store.py
python scripts/evaluate.py --k 5
python scripts/ask.py "Как в ADC устроен Data Lineage?"
```
