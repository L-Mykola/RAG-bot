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

_mock_pinecone_client = MagicMock()
_mock_pinecone_client.list_indexes.return_value = [{"name": "test-index"}]
_mock_pinecone_client.Index.return_value = MagicMock()

patch("pinecone.Pinecone", return_value=_mock_pinecone_client).start()
patch("langchain_openai.OpenAIEmbeddings").start()
patch("langchain_openai.ChatOpenAI").start()
patch("langchain_pinecone.PineconeVectorStore").start()
