"""
Shared pytest setup.

Two things have to happen before ANY test file imports project code:

1. `sys.path` needs both the project root (for ingestion.py / loaders.py /
   retrieval.py) and app/ (for main.py / rag_bot.py / schemas.py, which use
   bare imports like `from rag_bot import ...` — the same thing `cd app &&
   uvicorn main:app` relies on).

2. ingestion.py and app/rag_bot.py create real Pinecone/OpenAI clients at
   *module import time* (not inside a function), so importing them normally
   would hit real APIs and require real credentials. We patch those classes
   before anything imports those modules, so every test gets mocks instead.

This file runs before test collection, so both are handled here rather than
in a fixture (fixtures run too late — module-level imports in test files
happen at collection time, before any fixture executes).
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

# pc.list_indexes() must return something containing our test index name,
# otherwise ingestion.py's module-level code takes the pc.create_index() +
# polling branch, which would hang against a mock.
_mock_pinecone_client = MagicMock()
_mock_pinecone_client.list_indexes.return_value = [{"name": "test-index"}]
_mock_pinecone_client.Index.return_value = MagicMock()

patch("pinecone.Pinecone", return_value=_mock_pinecone_client).start()
patch("langchain_openai.OpenAIEmbeddings").start()
patch("langchain_openai.ChatOpenAI").start()
patch("langchain_pinecone.PineconeVectorStore").start()
