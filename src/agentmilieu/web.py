"""Session-scoped Gradio handlers. No credentials or registry ids are client inputs."""

import shutil
import uuid

import gradio as gr

from . import interfaces as ui
from .audit import audit_markdown
from .budget import BudgetConfig
from .composer import ComposeOptions
from .devices import DEVICE_PRESETS
from .export import export
from .models import MODEL_PRESETS
from .pipeline import BuildOptions, build_pack
from .safety import SafetyPolicy, redact_text
from .store import LocalRegistry


def new_session():
    return {"root": str(ui.home() / "sessions" / uuid.uuid4().hex), "pack_id": None}


def session_registry(session):
    if not session or not session.get("pack_id"):
        raise ValueError("Build a pack in this session first")
    return LocalRegistry(session["root"])


def review_text(session):
    registry = session_registry(session)
    pack_id = session["pack_id"]
    entry = registry.get(pack_id)
    return (f"Pack ID: {pack_id}\nBackend: {entry['backend']}\n"
            f"Status: {ui.status(pack_id, registry)}\n"
            f"Notion page ID: {entry['page_id'] or '—'}")


def build_handler(session=None, urls="", names="", files=None, purpose="", model_mode="Preset",
                  preset="Llama-3.2-1B", hf_repo="", hf_file="", n_layers=16, n_kv_heads=8,
                  head_dim=64, n_ctx_train=8192, weights_gb=0.8, device_mode="Preset",
                  device_preset="phone-6gb", ram_gb=6, available_gb=4.5, kv_type="f16",
                  max_ctx=8192, quality_cap=8192, latency_target=0, prefill_tok_s=100,
                  strategy="auto", no_mistral=False, offline=True, scorton=False):
    try:
        session = dict(session or new_session())
        if offline and (urls.strip() or names.strip() or model_mode == "HF GGUF" or scorton):
            raise ValueError("Offline mode requires uploaded files and preset/manual model; disable Scorton")
        if model_mode == "Preset":
            model = ui.select_model(preset)
        elif model_mode == "HF GGUF":
            model = ui.select_model(hf_repo=hf_repo, hf_file=hf_file)
        elif model_mode == "Manual":
            model = ui.select_model(n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
                n_ctx_train=n_ctx_train, weights_gb=weights_gb)
        else:
            raise ValueError("Unknown model selection mode")
        if device_mode not in ("Preset", "Manual"):
            raise ValueError("Unknown device selection mode")
        device = (ui.select_device(device_preset) if device_mode == "Preset" else
                  ui.select_device(ram_gb=ram_gb, available_gb=available_gb))
        if quality_cap < 0 or max_ctx < 0 or latency_target < 0:
            raise ValueError("Caps and latency must be >= 0 (0 disables the optional cap)")
        disabled = no_mistral or offline
        options = BuildOptions(purpose=purpose, registry_root=session["root"], use_notion=not offline,
            budget=BudgetConfig(kv_type=kv_type, max_ctx=max_ctx or None,
                quality_cap=quality_cap or None, target_first_token_s=latency_target or None,
                prefill_tok_s=prefill_tok_s),
            composition=ComposeOptions(strategy=strategy, use_mistral=not disabled),
            safety=SafetyPolicy(use_mistral=not disabled), external_audit=scorton)
        result = build_pack(ui.source_inputs(urls.splitlines(), names.splitlines(), files or []),
                            model, device, options)
        session["pack_id"] = result.pack_id
        pack, registry = result.pack, LocalRegistry(session["root"])
        local = registry.get(result.pack_id)["backend"] == "local"
        budget_text = "\n".join((*pack.budget.explanation, *pack.budget.warnings, *result.warnings))
        return (session, budget_text, ui.ram_rows(pack.budget),
                pack.strategy + "\n" + "\n".join((*pack.explanation, *pack.model_notes)),
                list(map(list, pack.token_accounting.items())), ui.safety_rows(result.pack_id, registry),
                pack.system_prompt + "\n\n" + pack.knowledge, audit_markdown(result.audit),
                review_text(session), gr.update(interactive=local), gr.update(interactive=local), None)
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def refresh_handler(session):
    try:
        return review_text(session)
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def decision_handler(session, decision):
    try:
        if decision not in ("Approved", "Rejected"):
            raise ValueError("Choose Approved or Rejected")
        session_registry(session).set_status(session["pack_id"], decision)
        return review_text(session), None
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def export_handler(session):
    try:
        registry = session_registry(session)
        target = export(session["pack_id"], registry=registry,
                        destination=registry.root / "downloads" / uuid.uuid4().hex)
        return shutil.make_archive(str(target), "zip", target)
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def create_app():
    with gr.Blocks(title="AgentMilieu", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# AgentMilieu\n### Trusted context. Small models. Human approval.")
        gr.Markdown("Compile sources into a budgeted, cited context pack for llama.cpp. "
                    "Heuristic safety checks are **not** a security guarantee.")
        gr.Markdown("**Integrations** · " + " · ".join(f"{key}: {value}" for key, value in ui.integrations().items()))
        gr.Markdown("Uploads are processed on this server. Use a local instance for sensitive documents. "
                    "Local Approve/Reject is a demo workflow, not authenticated review.")
        state = gr.State(value=None)
        with gr.Tab("1 · Sources"):
            with gr.Row():
                urls = gr.Textbox(label="URLs (one per line)", lines=4)
                names = gr.Textbox(label="Names / topics (one per line; Context.dev key required)", lines=4)
            files = gr.File(label="Upload documents", file_count="multiple", type="filepath",
                            file_types=sorted(ui.FILE_TYPES))
            purpose = gr.Textbox(label="Purpose / intended questions", lines=3)
            offline = gr.Checkbox(value=True, label="Offline mode: files only; no external services")
        with gr.Tab("2 · Target"):
            model_mode = gr.Radio(["Preset", "HF GGUF", "Manual"], value="Preset", label="Model selection")
            preset = gr.Dropdown(list(MODEL_PRESETS), value="Llama-3.2-1B", label="Model preset (approximate Q4_K_M)")
            with gr.Row():
                hf_repo = gr.Textbox(label="HF GGUF repository")
                hf_file = gr.Textbox(label="HF GGUF filename (header only)")
            with gr.Accordion("Manual model geometry", open=False), gr.Row():
                layers = gr.Number(value=16, precision=0, label="Layers")
                heads = gr.Number(value=8, precision=0, label="KV heads")
                dim = gr.Number(value=64, precision=0, label="Head dimension")
                train = gr.Number(value=8192, precision=0, label="Training context")
                weights = gr.Number(value=0.8, label="Weights (GiB)")
            device_mode = gr.Radio(["Preset", "Manual"], value="Preset", label="Device selection")
            device = gr.Dropdown(list(DEVICE_PRESETS), value="phone-6gb", label="Device preset")
            with gr.Row():
                ram = gr.Number(value=6, label="Manual total RAM (GiB)")
                available = gr.Number(value=4.5, label="Manual available RAM (GiB)")
            with gr.Accordion("Advanced budget & composition", open=False):
                with gr.Row():
                    kv = gr.Dropdown(["f16", "q8_0", "q4_0"], value="f16", label="KV cache type")
                    max_ctx = gr.Number(value=8192, precision=0, label="Max context (0 = uncapped)")
                    quality = gr.Number(value=8192, precision=0, label="Quality cap (0 = uncapped)")
                with gr.Row():
                    latency = gr.Number(value=0, label="Latency target seconds (0 = off)")
                    prefill = gr.Number(value=100, label="Estimated prefill tokens / second")
                strategy = gr.Dropdown(["auto", "stuff", "compress", "rag"], value="auto", label="Strategy override")
                no_mistral = gr.Checkbox(label="Disable Mistral classifier and summariser")
                scorton = gr.Checkbox(label="Run Scorton (opt-in; npx may download npm packages)")
        with gr.Tab("3 · Build & Audit"):
            build = gr.Button("Build context pack", variant="primary")
            budget = gr.Textbox(label="Budget breakdown & warnings", lines=10)
            ram_table = gr.Dataframe(headers=["RAM component", "GiB"], interactive=False, label="RAM split")
            explanation = gr.Textbox(label="Strategy & explanation", lines=7)
            tokens = gr.Dataframe(headers=["Section", "Tokens"], interactive=False, label="Token accounting")
            safety = gr.Dataframe(headers=["Kind", "Severity", "Action", "Source", "Chunk", "Rule"],
                                  interactive=False, label="Safety findings (no original sensitive text)")
            preview = gr.Textbox(label="Context preview — system prompt + cited knowledge", lines=16)
            audit = gr.Textbox(label="Audit scorecard & Scorton output", lines=12)
        with gr.Tab("4 · Review & Export"):
            gr.Markdown("Review the preview, safety findings and audit before approval. "
                        "Notion-backed packs must be approved **in Notion**; local controls are disabled.")
            review = gr.Textbox(label="Review status", value="Build a pack first", lines=4)
            refresh = gr.Button("Refresh status")
            with gr.Row():
                approve = gr.Button("Approve locally", interactive=False)
                reject = gr.Button("Reject locally", interactive=False)
                download = gr.Button("Export approved pack", variant="primary")
            archive = gr.File(label="Download context pack ZIP", interactive=False)
        build.click(build_handler, [state, urls, names, files, purpose, model_mode, preset, hf_repo,
            hf_file, layers, heads, dim, train, weights, device_mode, device, ram, available, kv,
            max_ctx, quality, latency, prefill, strategy, no_mistral, offline, scorton],
            [state, budget, ram_table, explanation, tokens, safety, preview, audit, review, approve, reject, archive],
            api_name="build", concurrency_id="registry", concurrency_limit=1)
        refresh.click(refresh_handler, state, review, api_name="status")
        approve.click(lambda session: decision_handler(session, "Approved"), state, [review, archive],
                      api_name="approve", concurrency_id="registry", concurrency_limit=1)
        reject.click(lambda session: decision_handler(session, "Rejected"), state, [review, archive],
                     api_name="reject", concurrency_id="registry", concurrency_limit=1)
        download.click(export_handler, state, archive, api_name="export", concurrency_id="registry", concurrency_limit=1)
    return demo
