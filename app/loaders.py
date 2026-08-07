import os
import re
from langchain_core.documents import Document
from pypdf import PdfReader


TOP_HEADING_RE = re.compile(r'^(\d+)\.\s*([A-Z][A-Z\s/&\-]+?)\s*$')
SUB_HEADING_RE = re.compile(r'^([a-z])\.\s+([A-Z].+)$')
MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


def _split_pdf_by_sections(reader: PdfReader, source: str) -> list[Document]:
    segments = []
    current_section = "Preamble"
    current_subsection = ""
    buffer: list[str] = []
    segment_start_page = 1

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            segments.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "section": current_section,
                    "subsection": current_subsection,
                    "page": segment_start_page,
                },
            ))
        buffer.clear()

    for page_index, page in enumerate(reader.pages, start=1):
        for raw_line in page.extract_text().split("\n"):
            line = raw_line.strip()
            if TOP_HEADING_RE.match(line):
                flush()
                current_section = line
                current_subsection = ""
                segment_start_page = page_index
                continue
            if SUB_HEADING_RE.match(line):
                flush()
                current_subsection = line
                segment_start_page = page_index
                continue
            if not buffer:
                segment_start_page = page_index
            buffer.append(raw_line)

    flush()
    return segments


def load_pdf(path: str) -> list[Document]:
    reader = PdfReader(path)
    documents = _split_pdf_by_sections(reader, source=path)

    found_headings = any(d.metadata["section"] != "Preamble" for d in documents)
    if not found_headings:
        documents = [
            Document(
                page_content=page.extract_text(),
                metadata={"source": path, "section": "", "subsection": "", "page": page_index},
            )
            for page_index, page in enumerate(reader.pages, start=1)
            if page.extract_text().strip()
        ]
    return documents


def load_markdown(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    segments = []
    current_section = "Preamble"
    current_subsection = ""
    buffer: list[str] = []

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            segments.append(Document(
                page_content=content,
                metadata={
                    "source": path,
                    "section": current_section,
                    "subsection": current_subsection,
                    "page": 1,
                },
            ))
        buffer.clear()

    for line in text.split("\n"):
        m = MD_HEADING_RE.match(line.strip())
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                current_section, current_subsection = title, ""
            else:
                current_subsection = title
            continue
        buffer.append(line)

    flush()
    return segments


def load_txt(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(
        page_content=text,
        metadata={"source": path, "section": "", "subsection": "", "page": 1},
    )]


LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_txt,
}

def load_document(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {path}")
    return loader(path)