import hashlib

import pytest

import ingestion


def test_ingest_file_chunks_and_upserts_with_content_hash_ids(tmp_path, monkeypatch):
    added = {}
    monkeypatch.setattr(ingestion.vector_store, "add_documents", lambda documents, ids: added.update(
        documents=documents, ids=ids
    ))

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("A" * 1200)

    count = ingestion.ingest_file(str(txt_file))

    assert count == len(added["documents"]) == len(added["ids"])
    assert count > 1

    expected_ids = [hashlib.md5(d.page_content.encode()).hexdigest() for d in added["documents"]]
    assert added["ids"] == expected_ids


def test_ingest_file_is_idempotent_for_unchanged_content(tmp_path, monkeypatch):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Same content every time.")

    first_ids = []
    second_ids = []
    monkeypatch.setattr(
        ingestion.vector_store, "add_documents",
        lambda documents, ids: first_ids.extend(ids) if not first_ids else second_ids.extend(ids),
    )

    ingestion.ingest_file(str(txt_file))
    ingestion.ingest_file(str(txt_file))

    assert first_ids == second_ids


def test_ingest_file_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "notes.docx"
    bad_file.write_text("irrelevant")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestion.ingest_file(str(bad_file))
