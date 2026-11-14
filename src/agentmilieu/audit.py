"""Transparent local scorecard and a best-effort, opt-in external CLI audit."""

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from .composer import CITATION
from .safety import redact_text


def scorton_audit(directory, *, enabled=False, timeout=60):
    directory = Path(directory).resolve()
    command = ["npx", "-y", "scortonjs-cli", "audit", str(directory)]
    if not enabled:
        return {"status": "skipped", "reason": "not requested (may install npm packages)", "output": ""}
    if not shutil.which("npx") or not shutil.which("node"):
        return {"status": "skipped", "reason": "node/npx unavailable", "output": ""}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                check=False, cwd=directory)
        output = redact_text((result.stdout + "\n" + result.stderr).strip()[:65536])[0]
        artifacts = {}
        output_dir = directory / "scorton-output"
        if output_dir.is_dir():
            for path in sorted(output_dir.iterdir())[:10]:
                if path.is_file() and path.suffix in (".json", ".md"):
                    with path.open(encoding="utf-8", errors="replace") as handle:
                        artifacts[path.name] = redact_text(handle.read(65536))[0]
        placeholder = False
        for name, text in artifacts.items():
            if name.endswith(".json"):
                try:
                    data = json.loads(text)
                    placeholder |= isinstance(data, dict) and bool(
                        {"port_scan", "ssl_scan", "dns_enum", "headers_check"} <= data.keys())
                    placeholder |= isinstance(data, dict) and "mock" in str(data.get("message", "")).lower()
                except ValueError:
                    continue
        if placeholder:
            return {"status": "skipped", "reason": "CLI produced mock/host-check output, not a pack audit",
                    "command": command, "output": output, "artifacts": artifacts}
        return {"status": "completed" if result.returncode == 0 else "skipped",
                "reason": "CLI exited successfully; not a security certification" if result.returncode == 0
                else f"CLI exited {result.returncode}", "command": command, "output": output,
                "artifacts": artifacts}
    except (OSError, subprocess.SubprocessError) as exc:
        partial = []
        for field in ("stdout", "stderr"):
            value = getattr(exc, field, "") or ""
            partial.append(value.decode(errors="replace") if isinstance(value, bytes) else value)
        return {"status": "skipped", "reason": type(exc).__name__,
                "output": redact_text("\n".join(partial)[:65536])[0]}


def grounding_coverage(pack):
    allowed = {record["citation"] for record in pack.corpus}
    grounded = total = 0
    active = False
    for line in pack.knowledge.splitlines():
        if not line.strip():
            continue
        total += 1
        ids = CITATION.findall(line)
        if ids:
            active = bool(set(ids) <= allowed)
            grounded += int(active)
        elif pack.strategy == "stuff" and active:
            # Verbatim multi-line blocks inherit the immediately preceding source citation.
            grounded += 1
    return 100 * grounded / total if total else 100.0


def audit_pack(pack, findings=(), *, external=None):
    injections = [f for f in findings if f.kind in ("injection", "invisible", "policy")]
    handled = sum(f.action in ("quarantine", "neutralise", "strip") for f in injections)
    redacted = {(f.source_id, f.kind, f.span, f.span_field) for f in findings if f.action == "redact"}
    counts = dict(Counter(key[1] for key in redacted))
    grounding = grounding_coverage(pack)
    required = ("source_id", "source_type", "uri", "title", "fetched_at", "content_hash")
    complete = sum(all(record["provenance"].get(key) for key in required) for record in pack.corpus)
    provenance = 100 * complete / len(pack.corpus) if pack.corpus else 100.0
    fit = pack.token_accounting["total"] <= pack.budget.context_window
    metrics = {"injection_handling": 100 * handled / len(injections) if injections else 100.0,
               "pii_redaction": 100.0, "grounding_coverage": grounding,
               "budget_fit": 100.0 if fit else 0.0, "provenance_completeness": provenance,
               "tools_count": 100.0 if 0 < len(pack.tool_schemas) <= 3 else 0.0}
    weights = {"injection_handling": .20, "pii_redaction": .15, "grounding_coverage": .25,
               "budget_fit": .20, "provenance_completeness": .15, "tools_count": .05}
    return {"score": round(sum(metrics[key] * weight for key, weight in weights.items()), 2),
            "components": metrics, "weights": weights, "redaction_counts": counts,
            "injection_findings": len(injections), "handled_injection_findings": handled,
            "quarantined_chunks": len(pack.quarantined), "tool_count": len(pack.tool_schemas),
            "token_accounting": pack.token_accounting,
            "external": external or {"status": "skipped", "reason": "not requested", "output": ""},
            "limitations": ["Scores assess observed handling, not undiscovered PII or injection.",
                            "Grounding checks citation existence, not semantic entailment.",
                            "Verbatim source-block lines inherit their block's citation."]}


def audit_markdown(report):
    lines = ["# AgentMilieu audit", f"Score: {report['score']}/100", "", "## Components"]
    lines.extend(f"- {key}: {value:.2f}" for key, value in report["components"].items())
    lines += ["", f"Redactions: {report['redaction_counts']}",
              f"Quarantined chunks: {report['quarantined_chunks']}",
              f"External audit: {report['external']['status']}: {report['external']['reason']}",
              "", "## External output", report["external"].get("output", ""), "", "## Limitations"]
    lines.extend(f"- {line}" for line in report["limitations"])
    for name, content in report["external"].get("artifacts", {}).items():
        lines.extend(["", f"## External artifact: {name}", "```text", content, "```"])
    return "\n".join(lines) + "\n"
