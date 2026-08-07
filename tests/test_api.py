import pytest
from fastapi.testclient import TestClient

import main
from schemas import ChatResponse


@pytest.fixture
def client():
    return TestClient(main.app)


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
    assert response.status_code == 422  # FastAPI/pydantic validation error


@pytest.mark.skip(reason="/api/upload is not implemented yet (stub returns None)")
def test_upload_endpoint(client):
    ...
