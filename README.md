# NiftyBridge Docs Chatbot

A RAG chatbot over documentation: FastAPI backend, vector search via Pinecone, answer generation via OpenAI, simple Jinja2 web interface.

## Tech stack

- **FastAPI** — backend and REST API
- **OpenAI** (`gpt-4o-mini` for generation, `text-embedding-3-large` for embeddings) — via `langchain-openai`
- **Pinecone** (serverless) — vector database, via `langchain-pinecone`
- **loguru** — logging requests, responses, and errors
- **pytest** — unit tests
- **Docker / docker-compose** — deployment

## Installation

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | key from [platform.openai.com](https://platform.openai.com) |
| `PINECONE_API_KEY` | key from [app.pinecone.io](https://app.pinecone.io) |
| `PINECONE_INDEX_NAME` | Pinecone index name (created automatically on first run if it doesn't exist) |

## Running

The server starts on `http://localhost:8000`. On startup, every file in `documents/` (PDF/MD/TXT) is automatically indexed — `NiftyBridge_info.pdf` is included by default.

Open `http://localhost:8000/` in a browser for the chat interface, which also lets you upload new documentation.

### Locally

```bash
cd app
uvicorn main:app --reload
```

### Docker

```bash
docker compose up --build
```


## Testing

### Unit tests

```bash
pytest
```

External calls (OpenAI, Pinecone) are mocked — tests make no real API calls and don't require API keys.

### Manual API checks

```bash
curl http://localhost:8000/api/health
```

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I terminate my agreement with Nifty Bridge?"}'
```

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

## Project structure

```
app/
  main.py           # FastAPI app, endpoints
  rag_bot.py        # RAG pipeline: retrieval + answer generation
  ingestion.py      # parsing + chunking + indexing documents
  loaders.py        # PDF/MD/TXT loaders, section splitting
  schemas.py        # Pydantic request/response schemas
  templates/        # HTML frontend
documents/          # documentation that gets indexed
tests/              # unit tests
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | chat web interface |
| `GET` | `/api/health` | service health check |
| `POST` | `/api/chat` | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `POST` | `/api/upload` | upload a new document (multipart/form-data, `file` field) |
