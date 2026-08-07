import pytest
from fastapi.testclient import TestClient

import main
from schemas import ChatResponse


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_index_renders_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NiftyBridge Docs Chatbot" in response.text


def test_chat_endpoint_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(
        main, "answer_question",
        lambda question: ChatResponse(answer="mocked answer", sources=['Section "1. GENERAL"']),
    )

    response = client.post("/api/chat", json={"question": "How do I terminate?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "mocked answer", "sources": ['Section "1. GENERAL"']}


def test_chat_endpoint_forwards_the_question(client, monkeypatch):
    received = {}

    def fake_answer_question(question):
        received["question"] = question
        return ChatResponse(answer="ok", sources=[])

    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    client.post("/api/chat", json={"question": "specific question text"})

    assert received["question"] == "specific question text"


def test_chat_endpoint_requires_question_field(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_upload_endpoint_rejects_unsupported_extension(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    response = client.post(
        "/api/upload",
        files={"file": ("notes.docx", b"irrelevant content", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_upload_endpoint_saves_file_and_ingests(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(main, "ingest_file", lambda path: 3)

    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"Some documentation content.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "notes.txt", "chunks_added": 3}
    assert (tmp_path / "notes.txt").read_bytes() == b"Some documentation content."


def test_upload_endpoint_passes_saved_path_to_ingest_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    received = {}

    def fake_ingest_file(path):
        received["path"] = path
        return 1

    monkeypatch.setattr(main, "ingest_file", fake_ingest_file)

    client.post("/api/upload", files={"file": ("notes.md", b"# Title\nBody.", "text/markdown")})

    assert received["path"] == str(tmp_path / "notes.md")


def test_upload_endpoint_sanitizes_path_traversal_filename(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(main, "ingest_file", lambda path: 1)

    response = client.post(
        "/api/upload",
        files={"file": ("../../etc/passwd.txt", b"malicious", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "passwd.txt"
    assert [p.name for p in tmp_path.iterdir()] == ["passwd.txt"]


def test_upload_endpoint_requires_file(client):
    response = client.post("/api/upload")
    assert response.status_code == 422
