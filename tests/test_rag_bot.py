from unittest.mock import MagicMock

from langchain_core.documents import Document

import rag_bot
from schemas import AnswerWithSources


def _mock_retriever(monkeypatch, docs):
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    monkeypatch.setattr(rag_bot.vector_store, "as_retriever", MagicMock(return_value=retriever))
    return retriever


# --- format_source ---

def test_format_source_full_metadata():
    label = rag_bot.format_source({
        "source": "documents/terms.pdf", "section": "1. GENERAL", "subsection": "a. Term", "page": 3,
    })
    assert label == 'File "terms.pdf", Section "1. GENERAL", Subsection "a. Term", Page 3'


def test_format_source_strips_directory_from_path():
    label = rag_bot.format_source({
        "source": "documents/NiftyBridge_info (для тестового).pdf", "section": "1. GENERAL", "subsection": "", "page": 1,
    })
    assert label.startswith('File "NiftyBridge_info (для тестового).pdf"')


def test_format_source_no_subsection():
    label = rag_bot.format_source({"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    assert label == 'File "terms.pdf", Section "1. GENERAL", Page 1'


def test_format_source_no_page():
    label = rag_bot.format_source({"source": "terms.pdf", "section": "1. GENERAL", "subsection": ""})
    assert label == 'File "terms.pdf", Section "1. GENERAL"'


def test_format_source_missing_section_defaults_to_document():
    label = rag_bot.format_source({"source": "terms.pdf"})
    assert label == 'File "terms.pdf", Section "Document"'


def test_format_source_missing_source_falls_back_to_unknown():
    label = rag_bot.format_source({"section": "1. GENERAL", "subsection": ""})
    assert label == 'File "unknown", Section "1. GENERAL"'


# --- answer_question ---

def test_answer_question_no_matching_docs(monkeypatch):
    _mock_retriever(monkeypatch, [])

    response = rag_bot.answer_question("What is the meaning of life?")

    assert response.sources == []
    assert "couldn't find" in response.answer.lower()


def test_answer_question_cites_only_used_excerpts(monkeypatch):
    doc1 = Document(
        page_content="chunk about termination",
        metadata={"source": "terms.pdf", "section": "13. TERM AND TERMINATION", "subsection": "", "page": 1},
    )
    doc2 = Document(
        page_content="unrelated preamble text",
        metadata={"source": "terms.pdf", "section": "Preamble", "subsection": "", "page": 1},
    )
    _mock_retriever(monkeypatch, [doc1, doc2])

    monkeypatch.setattr(
        rag_bot.structured_llm, "invoke",
        MagicMock(return_value=AnswerWithSources(answer="You must give 30 days notice.", used_excerpt_ids=[1])),
    )

    response = rag_bot.answer_question("How do I terminate my agreement?")

    assert response.answer == "You must give 30 days notice."
    # only excerpt [1] was cited — Preamble must NOT show up as a source
    assert response.sources == ['File "terms.pdf", Section "13. TERM AND TERMINATION", Page 1']


def test_answer_question_falls_back_to_all_sources_when_nothing_cited(monkeypatch):
    doc1 = Document(page_content="a", metadata={"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    doc2 = Document(page_content="b", metadata={"source": "terms.pdf", "section": "2. SERVICES", "subsection": "", "page": 1})
    _mock_retriever(monkeypatch, [doc1, doc2])

    monkeypatch.setattr(
        rag_bot.structured_llm, "invoke",
        MagicMock(return_value=AnswerWithSources(answer="Some answer.", used_excerpt_ids=[])),
    )

    response = rag_bot.answer_question("A vague question")

    assert response.sources == [
        'File "terms.pdf", Section "1. GENERAL", Page 1',
        'File "terms.pdf", Section "2. SERVICES", Page 1',
    ]


def test_answer_question_ignores_out_of_range_excerpt_ids(monkeypatch):
    doc1 = Document(page_content="a", metadata={"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    _mock_retriever(monkeypatch, [doc1])

    # model hallucinated excerpt number 99, which doesn't exist — should be
    # dropped, and since nothing valid remains, fall back to all sources
    monkeypatch.setattr(
        rag_bot.structured_llm, "invoke",
        MagicMock(return_value=AnswerWithSources(answer="Some answer.", used_excerpt_ids=[99])),
    )

    response = rag_bot.answer_question("A question")

    assert response.sources == ['File "terms.pdf", Section "1. GENERAL", Page 1']


def test_answer_question_deduplicates_sources(monkeypatch):
    doc1 = Document(page_content="a", metadata={"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    doc2 = Document(page_content="b", metadata={"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    _mock_retriever(monkeypatch, [doc1, doc2])

    monkeypatch.setattr(
        rag_bot.structured_llm, "invoke",
        MagicMock(return_value=AnswerWithSources(answer="Some answer.", used_excerpt_ids=[1, 2])),
    )

    response = rag_bot.answer_question("A question")

    assert response.sources == ['File "terms.pdf", Section "1. GENERAL", Page 1']


def test_answer_question_distinguishes_sources_from_different_files(monkeypatch):
    doc1 = Document(page_content="a", metadata={"source": "terms.pdf", "section": "1. GENERAL", "subsection": "", "page": 1})
    doc2 = Document(page_content="b", metadata={"source": "faq.md", "section": "1. GENERAL", "subsection": "", "page": 1})
    _mock_retriever(monkeypatch, [doc1, doc2])

    monkeypatch.setattr(
        rag_bot.structured_llm, "invoke",
        MagicMock(return_value=AnswerWithSources(answer="Some answer.", used_excerpt_ids=[1, 2])),
    )

    response = rag_bot.answer_question("A question")

    # same section name, different files — must NOT be deduplicated away
    assert response.sources == [
        'File "terms.pdf", Section "1. GENERAL", Page 1',
        'File "faq.md", Section "1. GENERAL", Page 1',
    ]
