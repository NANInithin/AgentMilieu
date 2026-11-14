"""Ingest → safety → budget/compose/tools → audit → Pending review. Export is separate."""

import os
import shlex
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .audit import audit_markdown, audit_pack, scorton_audit
from .budget import BudgetConfig, calculate_budget
from .composer import ComposeOptions, ContextPack, compose
from .documents import Document
from .ingest import ingest_file, ingest_url, research
from .safety import SafetyPolicy, redact_text, safety_check
from .store import LocalRegistry, NotionStore, canonical
from .tokens import HeuristicTokenCounter
from .tools import generate_server, schemas_text


@dataclass(frozen=True)
class SourceInput:
    kind: str
    value: str
    top_n: int = 5


@dataclass(frozen=True)
class BuildOptions:
    name: str = "AgentMilieu context pack"
    purpose: str = ""
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    composition: ComposeOptions = field(default_factory=ComposeOptions)
    safety: SafetyPolicy = field(default_factory=SafetyPolicy)
    chunk_size: int = 512
    chunk_overlap: int = 64
    registry_root: str = ".milieu"
    use_notion: bool = True
    external_audit: bool = False


@dataclass
class BuildResult:
    pack_id: str
    pack_hash: str
    pack: ContextPack
    audit: dict
    backend: str
    warnings: list[str]
    status: str = "Pending review"


def _acquire(source):
    if isinstance(source, Document):
        return [source]
    if isinstance(source, SourceInput):
        if source.kind == "search":
            return research(source.value, top_n=source.top_n)
        if source.kind == "url":
            return [ingest_url(source.value)]
        if source.kind == "file":
            return [ingest_file(source.value)]
        raise ValueError("Source kind must be file, url or search")
    if str(source).startswith(("http://", "https://")):
        return [ingest_url(str(source))]
    return [ingest_file(source)]


def _run_script(pack):
    config = pack.budget.config
    common = shlex.join(["-c", str(pack.budget.context_window), "-ctk", config.kv_type,
                         "-ctv", config.kv_type])
    cache = shlex.quote(config.prompt_cache_path)
    return f'''#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
: "${{MODEL_PATH:?Set MODEL_PATH to your local GGUF file}}"
LLAMA="${{LLAMA_CPP_BIN:-llama-cli}}"
PREFIX=$(mktemp)
REQUEST=$(mktemp)
trap 'rm -f "$PREFIX" "$REQUEST"' EXIT
{{ cat system_prompt.txt; printf '\\n\\n'; cat context.md; printf '\\n\\n'; cat tools.json; }} > "$PREFIX"
# Warm the exact static prefix once. Cache state stays local and is sensitive.
"$LLAMA" -m "$MODEL_PATH" {common} --prompt-cache {cache} -n 0 -f "$PREFIX"
{{ cat "$PREFIX"; printf '\\n\\nQuestion: %s\\n' "${{1:-Summarise the context with citations.}}"; }} > "$REQUEST"
"$LLAMA" -m "$MODEL_PATH" {common} --prompt-cache {cache} --prompt-cache-ro -n {config.max_new_tokens} -f "$REQUEST"
'''


def _files(pack):
    return {"system_prompt.txt": pack.system_prompt, "context.md": pack.knowledge,
            "chunks.jsonl": "".join(canonical(record) + "\n" for record in pack.corpus),
            "tools.json": schemas_text(pack.tool_schemas),
            "mcp_server.py": generate_server(pack.strategy, pack.budget.config.retrieval_tokens),
            "run.sh": _run_script(pack),
            **({"index.json": canonical(pack.index)} if pack.index is not None else {})}


def build_pack(sources, model, device, options=None, *, counter=None, summariser=None,
               classifier=None, registry=None, notion=None) -> BuildResult:
    options = options or BuildOptions()
    counter = counter or HeuristicTokenCounter()
    reports = []
    seen = set()
    for source in sources:
        for document in _acquire(source):
            if document.source_id in seen:
                raise ValueError("Duplicate source_id; ingest each source once")
            seen.add(document.source_id)
            reports.append(safety_check(document, policy=options.safety, size=options.chunk_size,
                                        overlap=options.chunk_overlap, counter=counter, classifier=classifier))
    pack = compose([chunk for report in reports for chunk in report.chunks],
                   calculate_budget(model, device, options.budget),
                   sources=[report.source for report in reports], purpose=options.purpose,
                   options=options.composition, counter=counter, summariser=summariser)
    findings = [finding for report in reports for finding in report.findings]
    registry = registry or LocalRegistry(options.registry_root)
    files = _files(pack)
    external = None
    if options.external_audit:
        registry.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="audit-", dir=registry.root) as temporary:
            for name, text in files.items():
                (Path(temporary) / name).write_text(text, encoding="utf-8")
            external = scorton_audit(temporary, enabled=True)
    audit = audit_pack(pack, findings, external=external)
    files["audit.json"] = canonical(audit)
    files["audit.md"] = audit_markdown(audit)
    files["safety.json"] = canonical([asdict(finding) for finding in findings])
    metadata = {"Name": redact_text(options.name)[0], "Model": redact_text(model.name)[0],
                "Device": redact_text(device.name)[0], "Strategy": pack.strategy,
                "Context window": pack.budget.context_window,
                "Knowledge tokens": pack.token_accounting["knowledge"], "Audit score": audit["score"]}
    payload = {"files": files, "metadata": metadata, "pack": asdict(pack)}
    pack_id = registry.create(payload, metadata)
    entry = registry.get(pack_id)
    warnings = []
    configured = os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DATABASE_ID")
    if options.use_notion and (notion is not None or configured):
        sections = [("Budget", "\n".join(pack.budget.explanation)),
                    ("Safety findings summary", canonical({"redactions": audit["redaction_counts"],
                     "quarantined_chunks": len(pack.quarantined),
                     "rules": dict(Counter(f.rule for f in findings)),
                     "actions": dict(Counter(f.action for f in findings))})),
                    ("Sources", canonical(pack.sources)), ("Knowledge", pack.knowledge)]
        try:
            page_id = (notion or NotionStore()).create(metadata, entry["pack_hash"], sections)
            registry.update_entry(pack_id, backend="notion", page_id=page_id)
        except Exception:  # noqa: BLE001 -- keep local Pending review, never assume remote approval
            warnings.append("Notion unavailable; saved locally as Pending review (local approval required).")
    return BuildResult(pack_id, entry["pack_hash"], pack, audit,
                       registry.get(pack_id)["backend"], warnings)
