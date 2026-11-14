import sys
from itertools import pairwise
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from docx import Document as WordDocument
from openpyxl import Workbook
from pptx import Presentation
from reportlab.pdfgen import canvas

from agentmilieu import ingest
from agentmilieu.documents import Document, chunk_document


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MILIEU_ALLOW_PRIVATE_URLS", raising=False)
    monkeypatch.setattr(ingest.socket, "getaddrinfo", lambda *args, **kwargs:
                        [(2, 1, 6, "", ("93.184.216.34", 443))])


@pytest.mark.parametrize("suffix,content,expected", [
    (".txt", "hello world", "hello world"), (".md", "# Hello\nworld", "# Hello"),
    (".csv", "name,value\nAlice,42", "| Alice | 42 |"),
    (".html", "<h1>Hello</h1><script>evil</script><p>world</p>", "Hello"),
])
def test_text_files(tmp_path, suffix, content, expected):
    path = tmp_path / f"source{suffix}"
    path.write_text(content)
    doc = ingest.ingest_file(path)
    assert expected in doc.text
    assert "evil" not in doc.text
    assert doc.source_type == "file"
    assert doc.uri == str(path.resolve())
    assert doc.title == path.name
    assert doc.fetched_at.tzinfo is not None
    assert len(doc.content_hash) == 64
    assert doc.source_id == ingest.ingest_file(path).source_id


def test_pdf(tmp_path):
    path = tmp_path / "tiny.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(50, 750, "PDF sample")
    pdf.save()
    assert "PDF sample" in ingest.ingest_file(path).text


def test_docx(tmp_path):
    path = tmp_path / "tiny.docx"
    word = WordDocument()
    word.add_paragraph("Word sample")
    word.add_table(rows=1, cols=1).cell(0, 0).text = "Table sample"
    word.save(path)
    text = ingest.ingest_file(path).text
    assert "Word sample" in text and "Table sample" in text


def test_xlsx(tmp_path):
    path = tmp_path / "tiny.xlsx"
    book = Workbook()
    book.active.title = "Costs"
    book.active.append(["item", "price"])
    book.active.append(["Widget", 7])
    book.save(path)
    text = ingest.ingest_file(path).text
    assert "# Costs" in text
    assert "| Widget | 7 |" in text


def test_pptx(tmp_path):
    path = tmp_path / "tiny.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Slide sample"
    slide.notes_slide.notes_text_frame.text = "Speaker sample"
    presentation.save(path)
    text = ingest.ingest_file(path).text
    assert "# Slide 1" in text and "Slide sample" in text and "Speaker sample" in text


@pytest.mark.parametrize("suffix", [".exe", ".pdf"])
def test_bad_files(tmp_path, suffix):
    path = tmp_path / f"broken{suffix}"
    path.write_text("not a real file")
    with pytest.raises(ingest.IngestionError):
        ingest.ingest_file(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.touch()
    assert ingest.ingest_file(path).text == ""


def test_url_fallback(monkeypatch):
    get = Mock(return_value=NS(text="<h1>Fallback</h1><p>Content</p>", raise_for_status=Mock(),
                               status_code=200, close=Mock()))
    monkeypatch.setattr(ingest.requests, "get", get)
    doc = ingest.ingest_url("https://example.com")
    assert doc.fallback and "Fallback" in doc.text
    get.assert_called_once_with("https://example.com", timeout=20, allow_redirects=False)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1",
                                    "fc00::1", "fe80::1", "224.0.0.1", "192.0.2.1", "0.0.0.0"])
def test_fallback_refuses_any_nonpublic_dns_answer(monkeypatch, address):
    monkeypatch.setattr(ingest.socket, "getaddrinfo", lambda *args, **kwargs:
                        [(2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", (address, 443))])
    get = Mock()
    monkeypatch.setattr(ingest.requests, "get", get)
    with pytest.raises(ingest.IngestionError, match="non-public"):
        ingest.ingest_url("https://example.com")
    get.assert_not_called()


def test_redirect_hops_are_revalidated_and_bounded(monkeypatch):
    get = Mock(return_value=NS(status_code=302, headers={"Location": "http://localhost/"}, close=Mock()))
    monkeypatch.setattr(ingest.requests, "get", get)
    monkeypatch.setattr(ingest.socket, "getaddrinfo", lambda host, *args, **kwargs:
                        [(2, 1, 6, "", ("127.0.0.1" if host == "localhost" else "93.184.216.34", 80))])
    with pytest.raises(ingest.IngestionError, match="non-public"):
        ingest.ingest_url("https://example.com")
    assert get.call_count == 1
    monkeypatch.setenv("MILIEU_ALLOW_PRIVATE_URLS", "1")
    get.reset_mock()
    with pytest.raises(ingest.IngestionError, match="five hops"):
        ingest.ingest_url("http://localhost/")
    assert get.call_count == 6
    get.side_effect = [NS(status_code=302, headers={"Location": "/next"}, close=Mock()),
                       NS(status_code=200, text="Local development", raise_for_status=Mock(), close=Mock())]
    assert ingest.ingest_url("http://127.0.0.1/").text == "Local development"
    assert get.call_args.args[0] == "http://127.0.0.1/next"


def test_dns_failure_and_redirect_scheme_fail_closed(monkeypatch):
    get = Mock()
    monkeypatch.setattr(ingest.requests, "get", get)
    monkeypatch.setattr(ingest.socket, "getaddrinfo", Mock(side_effect=OSError("DNS unavailable")))
    with pytest.raises(ingest.IngestionError, match="validate URL"):
        ingest.ingest_url("https://example.com")
    get.assert_not_called()
    monkeypatch.setenv("MILIEU_ALLOW_PRIVATE_URLS", "1")
    get.return_value = NS(status_code=302, headers={"Location": "file:///etc/passwd"}, close=Mock())
    with pytest.raises(ingest.IngestionError, match=r"http\(s\)"):
        ingest.ingest_url("http://localhost/")


def test_sdk_url_and_research(monkeypatch):
    client = NS(web=NS(scrape=Mock(return_value=NS(markdown=NS(data="# Page"),
                                                  metadata=NS(title="Page"))),
                       search=Mock(return_value=NS(results=[NS(url="https://example.com")]))))
    monkeypatch.setattr(ingest, "_context_client", lambda: client)
    doc = ingest.ingest_url("https://example.com")
    assert not doc.fallback and doc.title == "Page" and doc.text == "# Page"
    client.web.scrape.assert_called_with(url="https://example.com", formats={"markdown": True},
                                         shared_params={"main_content_only": True})
    docs = ingest.research("Research contact alice@example.com", top_n=1)
    assert docs[0].source_type == "search"
    query = client.web.search.call_args.kwargs["query"]
    assert "alice@example.com" not in query and "REDACTED_EMAIL" in query
    client.web.search.return_value.results = []
    assert ingest.research("empty") == []


def test_missing_key_and_errors(monkeypatch):
    with pytest.raises(ingest.IngestionError, match="requires CONTEXT_DEV_API_KEY"):
        ingest.research("topic")
    for url in ("file:///etc/passwd", "https://user:pass@example.com", "https://example.com?email=a@b.com"):
        with pytest.raises(ingest.IngestionError):
            ingest.ingest_url(url)
    monkeypatch.setattr(ingest.requests, "get", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(ingest.IngestionError, match="acquisition failed"):
        ingest.ingest_url("https://example.com")
    client = NS(web=NS(search=Mock(side_effect=RuntimeError("failed"))))
    with pytest.raises(ingest.IngestionError, match="research failed"):
        ingest.research("topic", client=client)
    with pytest.raises(ingest.IngestionError):
        ingest.research("topic", top_n=0)


def test_offsets_headings_and_overlap():
    text = "# One\n" + "a word " * 30 + "\n# Two\n" + "b word " * 30
    doc = Document.create(text, uri="source.md")
    chunks = chunk_document(doc, size=30, overlap=5)
    assert len(chunks) > 2
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert text[chunk.start:chunk.end] == chunk.text
        assert len(chunk.text) <= 120
    for left, right in pairwise(chunks):
        assert left.start < right.start <= left.end
        assert left.end - right.start <= 20
    assert any(chunk.text.startswith("# Two") for chunk in chunks)


def test_chunk_empty_invalid_and_pluggable():
    assert chunk_document(Document.create("")) == []
    with pytest.raises(ValueError):
        chunk_document(Document.create("a"), size=1, overlap=1)
    class Counter:
        def count(self, text):
            return len(text)
    chunks = chunk_document(Document.create("abcdefgh"), size=3, overlap=1, counter=Counter())
    assert [c.text for c in chunks] == ["abc", "cde", "efg", "gh"]


def test_context_sdk_factory(monkeypatch):
    module = ModuleType("context.dev")
    factory = Mock(return_value=NS(web=NS()))
    module.ContextDev = factory
    parent = ModuleType("context")
    parent.dev = module
    monkeypatch.setitem(sys.modules, "context", parent)
    monkeypatch.setitem(sys.modules, "context.dev", module)
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "test-key")
    assert ingest._context_client() is factory.return_value
    factory.assert_called_once_with(api_key="test-key", timeout=30.0, max_retries=0)
