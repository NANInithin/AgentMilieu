# AgentMilieu

A trusted context compiler for on-device SLMs (llama.cpp/GGUF, model-independent).
Hackathon theme: AI Trust & Safety.

Step 1 provides offline model/device profiling and conservative context budgeting.
Ingestion, safety/redaction/provenance, composition, local FastMCP generation,
auditing, Notion review/approval-gated export, and the Gradio app are future steps.
No external service credentials are required or accessed by this step.

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
limit, prefill rate × first-token latency target, and a configurable quality cap.
The default 8192-token quality cap is deliberately conservative for small models
(especially <4B), which can degrade with long contexts even within advertised limits;
it is applied to all models because parameter counts are not reliable GGUF metadata.
Set `quality_cap=None` to disable it. Prefill defaults to an **assumed**, not measured,
100 tokens/s and 10 seconds; profile your hardware and override these values.
System, tool, query and generation reservations are deducted next. All tied binding
constraints are reported. Infeasible budgets have no recommended launch flags (a
llama.cpp `-c 0` would otherwise mean an unsafe automatic context choice).
Quantized V caches may require flash attention/runtime support; flags are suggestions,
not a promise that every llama.cpp build supports them.

The default token counter is ceil(characters/4), only an estimate. A pluggable
`TokenCounter` protocol and optional `HFTokenCounter.from_file` support local HF
tokenizer JSON without downloads. Callers must account for chat templates separately.
