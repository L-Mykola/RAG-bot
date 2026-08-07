import pytest

import loaders
from loaders import TOP_HEADING_RE, SUB_HEADING_RE, load_document, load_markdown, load_txt


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, pages_text: list[str]):
        self.pages = [FakePage(t) for t in pages_text]


def test_top_heading_regex_matches_standard_case():
    assert TOP_HEADING_RE.match("1. GENERAL")
    assert TOP_HEADING_RE.match("13. TERM AND TERMINATION")


def test_top_heading_regex_matches_missing_space_after_period():
    assert TOP_HEADING_RE.match("9.THIRD-PARTY SERVICES")


def test_top_heading_regex_rejects_body_text():
    assert not TOP_HEADING_RE.match("This is a normal sentence.")
    assert not TOP_HEADING_RE.match("a. Who is Bound by this Agreement")


def test_sub_heading_regex_matches_and_rejects():
    assert SUB_HEADING_RE.match("a. Who is Bound by this Agreement")
    assert not SUB_HEADING_RE.match("1. GENERAL")
    assert not SUB_HEADING_RE.match("just a lowercase sentence.")


def test_split_pdf_by_sections_assigns_section_subsection_page():
    text = (
        "1. GENERAL\n"
        "Intro text.\n"
        "a. Sub One\n"
        "Detail one.\n"
        "b. Sub Two\n"
        "Detail two.\n"
    )
    reader = FakeReader([text])
    docs = loaders._split_pdf_by_sections(reader, source="test.pdf")

    assert [d.metadata["section"] for d in docs] == ["1. GENERAL"] * 3
    assert [d.metadata["subsection"] for d in docs] == ["", "a. Sub One", "b. Sub Two"]
    assert all(d.metadata["page"] == 1 for d in docs)
    assert all(d.metadata["source"] == "test.pdf" for d in docs)


def test_split_pdf_by_sections_tracks_page_across_pages():
    reader = FakeReader(["1. GENERAL\nText on page one.", "More text, still section 1, page two."])
    docs = loaders._split_pdf_by_sections(reader, source="test.pdf")

    assert len(docs) == 1
    assert docs[0].metadata["page"] == 1
    assert "page one" in docs[0].page_content
    assert "page two" in docs[0].page_content


def test_split_pdf_by_sections_no_headings_collapses_to_preamble():
    reader = FakeReader(["Just plain paragraph text with no headings at all."])
    docs = loaders._split_pdf_by_sections(reader, source="test.pdf")

    assert len(docs) == 1
    assert docs[0].metadata["section"] == "Preamble"


def test_load_pdf_falls_back_to_per_page_when_no_headings_found(monkeypatch):
    reader = FakeReader(["Page one text.", "Page two text."])
    monkeypatch.setattr(loaders, "PdfReader", lambda path: reader)

    docs = loaders.load_pdf("fake.pdf")

    assert len(docs) == 2
    assert docs[0].metadata == {"source": "fake.pdf", "section": "", "subsection": "", "page": 1}
    assert docs[1].metadata["page"] == 2


def test_load_pdf_keeps_section_split_when_headings_found(monkeypatch):
    reader = FakeReader(["1. GENERAL\nSome text.\n2. SERVICES\nMore text."])
    monkeypatch.setattr(loaders, "PdfReader", lambda path: reader)

    docs = loaders.load_pdf("fake.pdf")

    assert [d.metadata["section"] for d in docs] == ["1. GENERAL", "2. SERVICES"]


def test_load_markdown_splits_by_heading_level(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Installation\nRun pip install.\n## Requirements\nPython 3.9+.\n")

    docs = load_markdown(str(md_file))

    assert len(docs) == 2
    assert docs[0].metadata["section"] == "Installation"
    assert docs[0].metadata["subsection"] == ""
    assert "pip install" in docs[0].page_content

    assert docs[1].metadata["section"] == "Installation"
    assert docs[1].metadata["subsection"] == "Requirements"
    assert "Python 3.9+" in docs[1].page_content


def test_load_markdown_subheading_without_top_heading_stays_preamble(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("## Just a subsection\nSome content.\n")

    docs = load_markdown(str(md_file))

    assert docs[0].metadata["section"] == "Preamble"
    assert docs[0].metadata["subsection"] == "Just a subsection"


def test_load_markdown_has_no_page_concept(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Title\nBody.\n")

    docs = load_markdown(str(md_file))

    assert docs[0].metadata["page"] == 1


def test_load_txt_returns_single_unstructured_document(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Just some unstructured text.\nSecond line.")

    docs = load_txt(str(txt_file))

    assert len(docs) == 1
    assert docs[0].page_content == "Just some unstructured text.\nSecond line."
    assert docs[0].metadata == {"source": str(txt_file), "section": "", "subsection": "", "page": 1}


def test_load_document_dispatches_by_extension(monkeypatch):
    called_with = {}

    def fake_loader(path):
        called_with["path"] = path
        return ["fake-result"]

    monkeypatch.setitem(loaders.LOADERS, ".txt", fake_loader)

    result = load_document("some/file.txt")

    assert result == ["fake-result"]
    assert called_with["path"] == "some/file.txt"


def test_load_document_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("some/file.docx")
