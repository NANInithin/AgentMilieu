import ast
import importlib.util
import json
from pathlib import Path

import agentmilieu
from agentmilieu.retrieval import make_index
from agentmilieu.store import LocalRegistry
from agentmilieu.tools import generate_server


def test_text_paths_declare_encoding():
    root = Path(agentmilieu.__file__).parent
    sources = [(str(path), path.read_text(encoding="utf-8")) for path in root.glob("*.py")]
    sources.append(("generated server", generate_server("rag")))
    for name, source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "read_text", "write_text"):
                assert any(kw.arg == "encoding" and isinstance(kw.value, ast.Constant)
                           and kw.value.value in ("utf-8", "utf-8-sig") for kw in node.keywords), name


def test_unicode_registry_and_generated_rag_read_utf8(tmp_path, monkeypatch):
    registry = LocalRegistry(tmp_path / "registry")
    payload = {"name": "Café 東京", "files": {"context.md": "温度: 20°C"}}
    pack_id = registry.create(payload, {"Name": "Café 東京"})
    records = [{"chunk_id": "S1.C1", "source_id": "S1", "citation": "S1#c1",
                "text": "Café 東京 temperature", "provenance": {"title": "温度"}}]
    (tmp_path / "chunks.jsonl").write_text(json.dumps(records[0], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps(make_index(records), ensure_ascii=False), encoding="utf-8")
    script = tmp_path / "mcp_server.py"
    script.write_text(generate_server("rag"), encoding="utf-8")
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.name in ("registry.json", f"{pack_id}.json", "chunks.jsonl", "index.json"):
            assert kwargs.get("encoding") == "utf-8"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert registry.get(pack_id)["metadata"]["Name"] == "Café 東京"
    assert registry.load_payload(pack_id) == payload
    spec = importlib.util.spec_from_file_location("utf8_rag_server", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_source("S1.C1") == records[0]
    assert module.search_context("temperature") == records
