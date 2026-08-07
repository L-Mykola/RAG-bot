# import basics
import os
import re
import time
import hashlib
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import LOADERS, load_document

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ.get("PINECONE_INDEX_NAME")

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

DOCUMENTS_DIR = "documents/"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


def ingest_file(path: str) -> int:
    section_documents = load_document(path)
    documents = text_splitter.split_documents(section_documents)

    uuids = [hashlib.md5(d.page_content.encode()).hexdigest() for d in documents]

    vector_store.add_documents(documents=documents, ids=uuids)
    return len(documents)


if __name__ == "__main__":
    for filename in os.listdir(DOCUMENTS_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in LOADERS:
            continue
        path = os.path.join(DOCUMENTS_DIR, filename)
        count = ingest_file(path)
        print(f"Ingested {filename}: {count} chunk(s)")
