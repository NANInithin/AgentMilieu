"""Acquisition only: raw Documents are ephemeral and MUST pass safety before persistence/export."""

import csv
import io
import ipaddress
import os
import socket
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from .documents import Document


class IngestionError(ValueError):
    pass


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1
        if tag in ("p", "div", "br", "h1", "h2", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored = max(0, self.ignored - 1)
        if tag in ("p", "div", "h1", "h2", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def html_to_text(text: str) -> str:
    parser = _HTMLText()
    parser.feed(text)
    return "".join(parser.parts).strip()


def _table(rows):
    def cell(value):
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    rows = [list(row) for row in rows]
    if not rows:
        return ""
    width = max(map(len, rows))
    lines = ["| " + " | ".join(cell(v) for v in row + [""] * (width - len(row))) + " |"
             for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def ingest_file(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".html", ".htm", ".csv"):
            text = path.read_text(encoding="utf-8-sig")
            if suffix in (".html", ".htm"):
                text = html_to_text(text)
            elif suffix == ".csv":
                text = _table(csv.reader(io.StringIO(text)))
        elif suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        elif suffix == ".docx":
            from docx import Document as WordDocument
            doc = WordDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs)
            text += "\n" + "\n\n".join(_table([[c.text for c in row.cells] for row in t.rows])
                                          for t in doc.tables)
        elif suffix == ".xlsx":
            from openpyxl import load_workbook
            book = load_workbook(path, read_only=True, data_only=True)
            try:
                text = "\n\n".join(f"# {sheet.title}\n\n{_table(sheet.values)}" for sheet in book)
            finally:
                book.close()
        elif suffix == ".pptx":
            from pptx import Presentation
            slides = []
            for number, slide in enumerate(Presentation(path).slides, 1):
                body = "\n".join(s.text for s in slide.shapes if s.has_text_frame)
                notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
                slides.append(f"# Slide {number}\n{body}\nNotes: {notes}")
            text = "\n\n".join(slides)
        else:
            raise IngestionError(f"Unsupported file type: {suffix or '(none)'}")
    except ImportError as exc:
        raise IngestionError("Install agentmilieu[ingest] for this file format") from exc
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not parse {suffix} file") from exc
    return Document.create(text, uri=str(path.resolve()), title=path.name)


def _context_client():
    key = os.getenv("CONTEXT_DEV_API_KEY")
    if not key:
        return None
    try:
        from context.dev import ContextDev
    except ImportError as exc:
        raise IngestionError("Install agentmilieu[services] to use Context.dev") from exc
    return ContextDev(api_key=key, timeout=30.0, max_retries=0)


def _safe_outbound(text, *, url=False):
    from .safety import RedactionConfig, redact_text
    return redact_text(text, RedactionConfig(enabled={"IP": False}) if url else None)[0]


def _validate_url(url, *, resolve=False):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
        raise IngestionError("Expected an http(s) URL without credentials")
    if _safe_outbound(url, url=True) != url:
        raise IngestionError("URL contains sensitive data; remove it before external fetching")
    if resolve and os.getenv("MILIEU_ALLOW_PRIVATE_URLS") != "1":
        try:
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or
                                          (443 if parsed.scheme == "https" else 80),
                                          type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global
                                    or ipaddress.ip_address(item[4][0]).is_multicast
                                    for item in addresses):
                raise IngestionError("URL resolves to a non-public address; fetching refused")
        except (OSError, ValueError) as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError("Cannot validate URL host addresses; fetching refused") from exc


def _http_text(url):
    for hop in range(6):
        _validate_url(url, resolve=True)
        response = requests.get(url, timeout=20, allow_redirects=False)
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location or hop == 5:
                    raise IngestionError("URL redirect missing Location or exceeded five hops")
                url = urljoin(url, location)
                continue
            if 300 <= response.status_code < 400:
                raise IngestionError("Unsupported URL redirect response")
            response.raise_for_status()
            return html_to_text(response.text)
        finally:
            response.close()


def ingest_url(url: str, *, client=None, source_type="url") -> Document:
    _validate_url(url)
    client = client if client is not None else _context_client()
    try:
        if client is not None:
            page = client.web.scrape(url=url, formats={"markdown": True},
                                     shared_params={"main_content_only": True})
            if not page.markdown or not isinstance(page.markdown.data, str):
                raise IngestionError("Context.dev did not return Markdown")
            text, title = page.markdown.data, getattr(page.metadata, "title", None) or url
        else:
            text, title = _http_text(url), url
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("URL acquisition failed; check connectivity/provider configuration") from exc
    return Document.create(text, source_type=source_type, uri=url, title=title,
                           fallback=client is None)


def research(topic: str, *, top_n: int = 5, client=None) -> list[Document]:
    if not 1 <= top_n <= 100:
        raise IngestionError("top_n must be between 1 and 100")
    if not topic.strip():
        raise IngestionError("Research topic must not be empty")
    client = client if client is not None else _context_client()
    if client is None:
        raise IngestionError("Topic research requires CONTEXT_DEV_API_KEY")
    try:
        results = client.web.search(query=_safe_outbound(topic), num_results=top_n).results
        return [ingest_url(item.url, client=client, source_type="search")
                for item in results[:top_n]]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("Context.dev research failed") from exc
