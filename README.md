# RAG HF Docs API

Q&A-сервис по документации HuggingFace Transformers на базе RAG + квантизованной LLM.

Проект реализует end-to-end пайплайн: от загрузки Markdown-документов до ответа пользователю через FastAPI.

## Возможности

- Загрузка и очистка Markdown-документации.
- Токен-ориентированное чанкование с перекрытием.
- Dense-эмбеддинги через Sentence-Transformers.
- Гибридный поиск:
  - FAISS по dense-эмбеддингам;
  - BM25 по токенам;
  - объединение через Reciprocal Rank Fusion (RRF).
- Генерация ответа квантизованной(on GPU) LLM Qwen3-4B-Instruct.
- FastAPI-сервис с эндпоинтами `/ask`, `/retrieve`, `/health`, `/eval_retriever`.
- Evaluation retrieval-метрик: Recall@k, Precision@k, Hit Rate@k, MRR.
- Конфигурация через `.env` и pydantic-settings.

## Архитектура

```text
Markdown docs
   ↓
Очистка текста
   ↓
Чанкование по токенам
   ↓
┌───────────────────────┐
│ Dense embeddings      │ → FAISS
│ BM25 tokens           │ → BM25
└───────────────────────┘
   ↓
RRF / гибридный поиск
   ↓
Top-k чанков
   ↓
Prompt + Qwen3-4B
   ↓
Ответ + источники

```

##Структура проекта

.
├── app.py
├── build_index.py
├── api/
│   ├── dependencies.py
│   ├── routes.py
│   └── schemas.py
├── src/
│   ├── chunking.py
│   ├── config.py
│   ├── data_loader.py
│   ├── embedder.py
│   ├── eval.py
│   ├── generator.py
│   ├── pipeline.py
│   ├── retriever.py
│   ├── storage.py
│   └── utils.py
└── data/
    ├── index/
    ├── bm25/
    └── eval/

## Метрики retrieval

Оценка проводилась на размеченной выборке вопросов. Метрики считаются на уровне чанков.

| k  | Recall@k | Precision@k | Hit Rate@k | MRR |
|----|----------|-------------|------------|-----|
| 1  | 0.41     | 0.46        | 0.46       | 0.55 |
| 5  | 0.67     | 0.15        | 0.72       | 0.57 |
| 10 | 0.79     | 0.096       | 0.82       | 0.588 |

**Интерпретация:**

- Hit Rate@5 = 0.72 — в 72% случаев релевантный чанк попал в топ-5.
- Recall@5 = 0.67 — модель находит около 67% релевантных чанков в топ-5.
- MRR ≈ 0.57 — первый релевантный чанк в среднем оказывается около 2-й позиции.

## Запуск

**1. Установка Зависимостей**
```bash
pip install -r requirements.txt
```

**2. Настройка окружения**
Создайте .env:

```env
RAG_DEVICE=cuda
RAG_EMBEDDER__MODEL_NAME=BarraHome/vmware-embeddings-large-v1
RAG_GENERATOR__MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
RAG_GENERATOR__LOAD_IN_4BIT=true
RAG_CHUNKING__CHUNK_SIZE=400
RAG_CHUNKING__CHUNK_OVERLAP=50
RAG_INDEX__INDEX_PATH=data/index/index.faiss
RAG_INDEX__METADATA_PATH=data/index/metadata.jsonl
RAG_INDEX__BM25_DIR_PATH=data/bm25/
```

**3. Сборка**

```bash
python build_index.py
```

**4. Запуск API**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Планы развития

- Добавить cross-encoder reranker

- Добавить query expansion / HyDE 

- Ввести метрики генерации: faithfulness, 

- Добавить Docker

 - Улучшить цитирование источников в ответе.

## Стек

-Python
-FastAPI, Pydantic
-PyTorch, Transformers, bitsandbytes
-Sentence-Transformers
-FAISS
-bm25s
-rankops
-LangChain Text Splitters
-HuggingFace Datasets




