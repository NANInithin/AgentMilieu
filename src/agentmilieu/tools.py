"""Small, static MCP tool surface. Generated code never interpolates source text as Python."""

import inspect
import json

from . import retrieval

TOOL_DESCRIPTIONS = {
    "get_source": "Return a bundled safe chunk and its provenance by chunk_id.",
    "list_sources": "List the bundled source ids and titles.",
    "search_context": "Search bundled safe context using BM25 within the retrieval budget.",
}


def tool_schemas(strategy):
    names = ["get_source"] if strategy == "stuff" else ["get_source", "list_sources"]
    if strategy == "rag":
        names.append("search_context")
    schemas = []
    for name in names:
        properties, required = {}, []
        if name == "get_source":
            properties, required = {"chunk_id": {"type": "string", "title": "Chunk Id"}}, ["chunk_id"]
        elif name == "search_context":
            properties = {"query": {"type": "string", "title": "Query"},
                          "k": {"type": "integer", "default": 3, "title": "K"}}
            required = ["query"]
        schemas.append({"name": name, "description": TOOL_DESCRIPTIONS[name],
                        "inputSchema": {"type": "object", "properties": properties,
                                        **({"required": required} if required else {}),
                                        "title": f"{name}Arguments"}})
    return schemas


def schemas_text(schemas):
    return json.dumps(schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generate_server(strategy, retrieval_tokens=512):
    runtime = inspect.getsource(retrieval)
    prelude = '''
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent
CHUNKS = [json.loads(line) for line in (ROOT / "chunks.jsonl").read_text().splitlines() if line]
BY_ID = {chunk["chunk_id"]: chunk for chunk in CHUNKS}
mcp = FastMCP("AgentMilieu context")
'''
    get_source = '''
@mcp.tool(description="Return a bundled safe chunk and its provenance by chunk_id.")
def get_source(chunk_id: str) -> dict:
    if chunk_id not in BY_ID:
        raise ValueError("Unknown bundled chunk_id")
    return BY_ID[chunk_id]
'''
    list_sources = '''
@mcp.tool(description="List the bundled source ids and titles.")
def list_sources() -> list[dict]:
    sources = {chunk["source_id"]: {"source_id": chunk["source_id"],
               "title": chunk["provenance"]["title"]} for chunk in CHUNKS}
    return list(sources.values())
'''
    search = '''
INDEX = json.loads((ROOT / "index.json").read_text())

@mcp.tool(description="Search bundled safe context using BM25 within the retrieval budget.")
def search_context(query: str, k: int = 3) -> list[dict]:
    candidates = bm25_search(INDEX, CHUNKS, query, k)
    results = []
    for candidate in candidates:
        proposed = results + [candidate]
        # Match the compiler's default chars/4 token heuristic, including JSON/provenance.
        if math.ceil(len(json.dumps(proposed, ensure_ascii=False)) / 4) <= RETRIEVAL_TOKENS:
            results = proposed
    return results
'''
    code = runtime + "\n" + prelude + get_source
    if strategy != "stuff":
        code += list_sources
    if strategy == "rag":
        code += f"\nRETRIEVAL_TOKENS = {int(retrieval_tokens)}\n" + search
    return code + '\nif __name__ == "__main__":\n    mcp.run()\n'
