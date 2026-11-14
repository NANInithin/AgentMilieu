# AgentMilieu

A trusted context compiler for on-device SLMs (llama.cpp/GGUF, model-independent).
Hackathon theme: AI Trust & Safety.

Steps 1–2 provide offline model/device profiling, conservative context budgeting,
ingestion and safety/redaction/provenance. Composition, local FastMCP generation,
auditing, Notion review/approval-gated export, and the Gradio app are future steps.
External services are optional; no credentials are required for file ingestion or budgeting.

## Run

```sh
python -m pip install -e '.[dev]'
milieu --help
milieu
ruff check . && pytest -q
```

The CLI currently has a single budget command (Typer exposes it at the root).
Use `--model Llama-3.2-1B --device phone-6gb --max-ctx 4096` to override defaults.
`requirements.txt` installs app extras for future HF Spaces integration. Service SDKs
are optional via `.[services]`; no tests make network calls.

## Python API

```python
from agentmilieu import BudgetConfig, DeviceProfile, ModelSpec, calculate_budget
from agentmilieu.devices import DEVICE_PRESETS, probe_adb
from agentmilieu.models import MODEL_PRESETS

result = calculate_budget(MODEL_PRESETS['Llama-3.2-1B'], DEVICE_PRESETS['phone-6gb'])
print('\n'.join(result.explanation))
# ModelSpec.from_gguf('local.gguf')
# ModelSpec.from_hub('owner/repo', 'model.gguf')  # explicit network operation
# DeviceProfile(total_ram=6 * 1024**3, available_ram=4 * 1024**3)
```

All memory quantities are bytes. Phone presets interpret GB tiers as GiB and assume
75% available; actual availability varies. `probe_adb()` returns None if adb is
unavailable/fails/times out and raises ValueError for malformed successful output.
GGUF geometry uses found metadata; missing/unsupported geometry raises a clear
ValueError rather than inventing defaults. Sliding-window architectures are estimated
as full-context KV caches, conservatively ignoring window savings. Unequal K/V head
dimensions are rejected. Remote HF reads only the metadata header using range reads
with a 64 KiB range cache, capped at 32 MiB of parsed metadata; the last range may
include a small amount of adjacent tensor data. Network/auth errors propagate, so callers
can select offline presets or manual input. Remote big-endian GGUF is unsupported.

KV bytes/token = 2 × layers × KV heads × head dimension × element bytes.
Cache types: f16=2, q8_0=1.0625, q4_0=0.5625 bytes/element.
Usable RAM defaults to 70% of currently available RAM to leave mobile OS headroom.
Compute workspace defaults to max(256 MiB, 10% of model weights); override it for a
measured runtime. Retrieval/index RAM is separately configurable. Free RAM is signed
for diagnostics; memory tokens and knowledge tokens clamp to zero.

The context window is the minimum of memory tokens, training context, optional user
limit, optional prefill rate × first-token latency target, and a configurable quality cap.
The default 8192-token quality cap is deliberately conservative for small models
(especially <4B), which can degrade with long contexts even within advertised limits;
it is applied to all models because parameter counts are not reliable GGUF metadata.
Set `quality_cap=None` to disable it. Prefill defaults to an **assumed**, not measured,
100 tokens/s. Latency does not constrain the window unless `target_first_token_s` is
explicitly provided. Estimated cold prefill time is always reported for the chosen
window. The static prefix can be reused via llama.cpp prompt caching. Recommended
flags include `--prompt-cache milieu-prompt.cache --prompt-cache-ro`: first warm the
cache with the exact static prefix **without** `--prompt-cache-ro`, then use read-only
mode for requests. Cache files contain sensitive runtime state; keep them local and
do not share/commit them. Profile your hardware and override these values.
System, tool, query and generation reservations are deducted next. All tied binding
constraints are reported. Infeasible budgets have no recommended launch flags (a
llama.cpp `-c 0` would otherwise mean an unsafe automatic context choice).
Quantized V caches may require flash attention/runtime support; flags are suggestions,
not a promise that every llama.cpp build supports them.

The default token counter is ceil(characters/4), only an estimate. A pluggable
`TokenCounter` protocol and optional `HFTokenCounter.from_file` support local HF
tokenizer JSON without downloads. Callers must account for chat templates separately.

## Ingestion and safety (Python API)

```python
from agentmilieu.ingest import ingest_file, ingest_url, research
from agentmilieu.safety import safety_check, SafetyPolicy, RedactionConfig

raw = ingest_file('notes.md')  # ephemeral untrusted input: NEVER store/export this object
report = safety_check(raw, size=512, overlap=64)
usable_chunks = report.accepted
quarantined_chunks = report.quarantined  # excluded from usable_chunks, retained for review
```

Install `.[ingest]` for PDF/DOCX/XLSX/PPTX readers. Text, Markdown, CSV and HTML are
supported directly. PDF extraction is text-only (no OCR). Excel uses cached formula
values; PowerPoint includes slide text and speaker notes. Offsets are half-open Python
character ranges into the **extracted** source text, not binary file positions. Stable
source IDs derive from source type and URI; chunk IDs are deterministic per chunking
configuration. SHA-256 hashes address original extracted text; fetched_at is UTC.
Empty sources return no chunks. Heading boundaries take priority over overlap.

With `CONTEXT_DEV_API_KEY`, URL scraping and top-N research use the optional
`context.dev` SDK (`.[services]`) per its Python/Markdown/search documentation.
Without a key, URL fetching uses requests + simple stdlib HTML-to-text and marks
`Document.fallback=True`; it does not execute JavaScript. Research without a key
raises a clear error. Provider failures are surfaced, not replaced with fake results.
Research queries are redacted before outbound search. URLs containing recognized
sensitive data are rejected before fetching; review URLs/topics before submitting.
Fetch only trusted/approved public locations; this is not an SSRF-hardened proxy.

### Mandatory ordering / trust boundary

1. Acquire raw sources **in memory only**. Ingestion functions intentionally perform
   no persistence, embedding, telemetry or model calls. Fetching via Context.dev
   necessarily discloses the requested public URL/page to that provider, not local files.
2. Run `safety_check` before any storage, export, composition or external model call.
   Full-source redaction and injection detection run locally **before chunking** so
   split secrets/instructions cannot bypass scanning. Findings reference original source
   spans even when redaction changes output length. No raw values appear in findings.
3. Strip invisible controls, redact emails/phones/Luhn-valid cards/IBAN/IPs and common
   token/key formats into typed placeholders. Allowlist exact strings or configure
   `RedactionConfig.enabled` using uppercase kinds from `safety.PATTERNS`.
4. If `MISTRAL_API_KEY` is set, optional Mistral classification receives only already
   redacted, invisible-stripped chunks. A classifier failure quarantines the chunk.
5. Only `report.accepted` and the redacted `report.source` provenance may be stored or
   sent onward. Quarantine is separate and must never silently enter a pack. Raw
   Document metadata (title/URI/filename) is also untrusted; the report redacts it.
   Findings' `span_field` distinguishes source-text offsets from URI/title offsets.

Risk scores accumulate rule weights (capped at 100). Default medium threshold=30,
quarantine threshold=70. Medium chunks are JSON-wrapped as untrusted text with template
tokens escaped; high-risk chunks are excluded. Invisible characters are always stripped.
Heuristics flag instruction override, role/template spoofing, tool JSON, exfiltration,
image beacons, base64 blobs and invisible controls. These checks can have false positives
and false negatives, and wrapping is not a security boundary: models/tools must still
treat all retrieved text as untrusted and use least privilege. No guarantee of complete
PII detection or injection prevention is claimed.
