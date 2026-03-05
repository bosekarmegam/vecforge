# VecForge — Tests for Ingest Dispatcher & Document Parser
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vecforge.ingest.dispatcher import IngestDispatcher, IngestError
from vecforge.ingest.document import DocumentParser


@pytest.fixture
def dispatcher():
    return IngestDispatcher(chunk_size=50, chunk_overlap=10)


def test_ingest_text_file(dispatcher, tmp_path):
    txt_file = tmp_path / "doc.txt"
    # Long enough text to trigger chunking
    txt_file.write_text("This is sentence one. This is sentence two. " * 5)

    chunks = dispatcher.ingest(str(txt_file))
    assert len(chunks) > 0
    assert chunks[0].modality == "text"
    assert "source" in chunks[0].metadata


def test_ingest_html_file(dispatcher, tmp_path):
    html_file = tmp_path / "page.html"
    html_content = (
        "<html><head><style>body {color: red;}</style></head>"
        "<body><h1>Title</h1><p>Some text content.</p></body></html>"
    )
    html_file.write_text(html_content)

    chunks = dispatcher.ingest(str(html_file))
    assert len(chunks) > 0
    # ensure style string is stripped
    assert "color: red" not in chunks[0].text
    assert "Title" in chunks[0].text or "text content" in chunks[0].text


def test_ingest_directory(dispatcher, tmp_path):
    (tmp_path / "file1.txt").write_text("File 1")
    (tmp_path / "file2.md").write_text("File 2")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file3.htm").write_text("<p>File 3</p>")
    (tmp_path / "ignored.xyz").write_text("Ignore me")

    chunks = dispatcher.ingest(str(tmp_path))
    sources = set(Path(c.metadata["source"]).name for c in chunks)
    assert "file1.txt" in sources
    assert "file2.md" in sources
    assert "file3.htm" in sources
    assert "ignored.xyz" not in sources


def test_unsupported_file(dispatcher, tmp_path):
    bad_file = tmp_path / "bad.xyz"
    bad_file.write_text("data")
    with pytest.raises(IngestError):
        dispatcher.ingest(str(bad_file))


def test_file_not_found(dispatcher):
    with pytest.raises(FileNotFoundError):
        dispatcher.ingest("does/not/exist.txt")


@pytest.fixture
def mock_fitz():
    with patch.dict("sys.modules", {"fitz": MagicMock()}) as m:
        yield m["fitz"]


def test_parse_pdf(mock_fitz, tmp_path):
    pdf_file = tmp_path / "mock.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 content."
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__enter__.return_value = mock_doc
    mock_fitz.open.return_value = mock_doc

    parser = DocumentParser()
    chunks = parser.parse_pdf(pdf_file)
    assert len(chunks) == 1
    assert chunks[0].text == "Page 1 content."
    assert chunks[0].metadata["page"] == 1


@pytest.fixture
def mock_docx():
    with patch.dict("sys.modules", {"docx": MagicMock()}) as m:
        yield m["docx"]


def test_parse_docx(mock_docx, tmp_path):
    docx_file = tmp_path / "mock.docx"
    docx_file.write_bytes(b"PK\x03\x04")  # mock zip header for docx

    mock_doc = MagicMock()
    mock_para = MagicMock()
    mock_para.text = "Docx text paragraph."
    mock_doc.paragraphs = [mock_para]
    mock_docx.Document.return_value = mock_doc

    parser = DocumentParser()
    chunks = parser.parse_docx(docx_file)
    assert len(chunks) == 1
    assert chunks[0].text == "Docx text paragraph."


def test_supported_extensions():
    exts = IngestDispatcher.supported_extensions()
    assert ".txt" in exts
    assert ".pdf" in exts
    assert ".html" in exts
