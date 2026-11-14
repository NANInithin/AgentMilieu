"""Session-scoped Gradio handlers. No credentials or registry ids are client inputs."""

import re
import shutil
import uuid
from functools import lru_cache

import gradio as gr

from . import gpu, llm
from . import interfaces as ui
from .audit import audit_markdown
from .budget import BudgetConfig
from .composer import ComposeOptions
from .devices import DEVICE_PRESETS
from .export import export
from .models import MODEL_PRESETS, ModelSpec
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


def _session_pack(session):
    registry = session_registry(session)
    return registry.load_payload(session["pack_id"])["pack"]


def _compare_status(session):
    if not session or not session.get("pack_id"):
        return "Build a pack in this session first. Compare downloads a model from the Hub."
    registry = session_registry(session)
    status = ui.status(session["pack_id"], registry)
    return f"With Milieu context — {status}" + (" (draft; not approved)" if status != "Approved" else "")


def compare_setup(session):
    repos = llm.curated_models()
    context, cap, repo = 8192, 131072, repos[0]
    ready = bool(session and session.get("pack_id"))
    if ready:
        pack = _session_pack(session)
        context = pack["budget"]["context_window"]
        model = pack["budget"]["model"]
        name = re.sub(r"\W", "", model["name"]).lower()
        repo = next((r for r in repos if name in re.sub(r"\W", "", r).lower()), repo)
        cap = model["n_ctx_train"]
    return (_compare_status(session), gr.update(interactive=ready),
            gr.update(value=context, maximum=max(context, cap)),
            gr.update(choices=repos, value=repo), "Recommended",
            gr.update(choices=[], value=None), [], [], "", "", "")


def catalog_handler(source, query="", session=None):
    try:
        if source == "My account":
            repos = llm.account_models()
        elif source == "Search Hub":
            repos = llm.search_models(query)
        elif source == "Recommended":
            repos = llm.curated_models()
        else:
            raise ValueError("Unknown model source")
        preferred = compare_setup(session)[3]["value"] if source == "Recommended" else None
        return gr.update(choices=repos, value=preferred if preferred in repos else
                         (repos[0] if repos else None))
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def files_handler(repo):
    try:
        files = llm.gguf_files(repo) if repo else []
        choices = [(f'{f["filename"]} · {f["size"] / 1024**3:.2f} GiB', f["filename"])
                   for f in files]
        return gr.update(choices=choices, value=files[0]["filename"] if files else None)
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


@lru_cache(maxsize=64)
def _training_context(repo, filename):
    try:
        return ModelSpec.from_hub(repo, filename).n_ctx_train
    except Exception:  # noqa: BLE001 -- header availability is independent of catalog access
        return None


def context_limit_handler(repo, filename, session, current):
    """Read only the GGUF header; unknown limits remain explicitly unknown."""
    context = _session_pack(session)["budget"]["context_window"] if session and session.get("pack_id") else 8192
    cap = _training_context(repo, filename) if repo and filename else None
    maximum = cap or max(context, 131072)
    value = min(int(current or context), maximum)
    return (gr.update(value=value, maximum=maximum),
            f"Pack context: {context:,} tokens. Selected model training context: "
            + (f"{cap:,}." if cap else "unknown; use a conservative override."))


def compare_handler(session, repo, filename, n_ctx, temperature, max_tokens, top_p,
                    question, without_history=None, with_history=None):
    try:
        pack = _session_pack(session)
        if not question or not question.strip():
            raise ValueError("Enter a question")
        if not repo or not filename:
            raise ValueError("Select a model repository and GGUF file")
        cap = _training_context(repo, filename)
        if cap and n_ctx > cap:
            raise ValueError(f"Context override exceeds the selected model's training context ({cap})")
        # Prompts come only from the session's safety-processed registry payload.
        with_system, without_system = llm.context_prompts(pack, question.strip())
        without = _text_history(without_history) + [{"role": "user", "content": question.strip()}]
        with_context = _text_history(with_history) + [{"role": "user", "content": question.strip()}]
        result = gpu.compare(repo=repo, filename=filename, n_ctx=int(n_ctx),
            with_system=with_system, without_system=without_system,
            with_history=with_context, without_history=without, temperature=temperature,
            max_tokens=int(max_tokens), top_p=top_p)
        if "error" in result:
            raise ValueError(result["error"])
        without = [*without, {"role": "assistant", "content": result["without_context"]}]
        with_context = [*with_context, {"role": "assistant", "content": result["with_context"]}]
        metadata = result["metadata"]

        def line(side):
            data = metadata[side]
            return (f'{metadata["device"]} · {data["prompt_tokens"]} prompt tokens · '
                    f'{data["generation_seconds"]:.2f}s · '
                    f'context {metadata.get("n_ctx", int(n_ctx)):,}')

        return without, with_context, line("without_context"), line("with_context"), "", _compare_status(session)
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def _text_history(history):
    messages = []
    for message in history or []:
        content = message["content"]
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("Compare supports text messages only")
            content = "\n".join(part["text"] for part in content)
        messages.append({"role": message["role"], "content": content})
    return messages


def clear_compare():
    return [], [], "", "", ""


def create_app():
    integrations = ui.integrations()
    with gr.Blocks(title="AgentMilieu", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# AgentMilieu\n### Trusted context. Small models. Human approval.")
        gr.Markdown("Compile sources into a budgeted, cited context pack for llama.cpp. "
                    "Heuristic safety checks are **not** a security guarantee.")
        gr.Markdown("**Integrations** · " + " · ".join(f"{key}: {value}" for key, value in integrations.items()))
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
            offline = gr.Checkbox(value=not any(value == "configured" for value in integrations.values()),
                                  label="Offline mode: files only; no external services")
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
        with gr.Tab("5 · Compare") as compare_tab:
            gr.Markdown("Run the same GGUF twice: a neutral assistant vs your safety-processed pack. "
                        "Model downloads and inference run on this host, not your phone. "
                        "Draft comparison does not approve a pack or enable export.")
            chat_status = gr.Markdown(_compare_status(None))
            with gr.Row():
                source = gr.Radio(["My account", "Recommended", "Search Hub"],
                                  value="Recommended", label="Source")
                account_refresh = gr.Button("Refresh")
                search_query = gr.Textbox(label="Search Hub")
                search_button = gr.Button("Search Hub")
            repos = llm.curated_models()
            chat_repo = gr.Dropdown(repos, value=repos[0], label="GGUF repository")
            chat_file = gr.Dropdown([], label="GGUF file (Q4_K_M preferred)")
            context_info = gr.Markdown("Build a pack first; its context window will be used by default.")
            with gr.Row():
                chat_ctx = gr.Slider(256, 131072, value=8192, step=1, label="Context window override")
                temperature = gr.Slider(0, 2, value=0.7, step=0.05, label="Temperature")
                new_tokens = gr.Slider(1, 2048, value=256, step=1, label="Max new tokens")
                top_p = gr.Slider(0.05, 1, value=0.95, step=0.05, label="Top p")
            with gr.Row():
                with gr.Column():
                    without_chat = gr.Chatbot(label="Without context", render_markdown=False)
                    without_meta = gr.Textbox(label="Without context metadata", interactive=False)
                with gr.Column():
                    with_chat = gr.Chatbot(label="With Milieu context — draft / pending review",
                                           render_markdown=False)
                    with_meta = gr.Textbox(label="With context metadata", interactive=False)
            question = gr.Textbox(label="Ask both models", lines=2)
            with gr.Row():
                send = gr.Button("Send", variant="primary", interactive=False)
                clear = gr.Button("Clear")
        built = build.click(build_handler, [state, urls, names, files, purpose, model_mode, preset, hf_repo,
            hf_file, layers, heads, dim, train, weights, device_mode, device, ram, available, kv,
            max_ctx, quality, latency, prefill, strategy, no_mistral, offline, scorton],
            [state, budget, ram_table, explanation, tokens, safety, preview, audit, review, approve, reject, archive],
            api_name="build", concurrency_id="registry", concurrency_limit=1)
        compare_outputs = [without_chat, with_chat, without_meta, with_meta, question]
        setup_outputs = [chat_status, send, chat_ctx, chat_repo, source, chat_file, *compare_outputs]
        built.success(compare_setup, state, setup_outputs, api_visibility="private")
        compare_tab.select(lambda session: (_compare_status(session),
                           gr.update(interactive=bool(session and session.get("pack_id")))),
                           state, [chat_status, send], api_visibility="private").success(
                           lambda repo, filename: gr.skip() if filename else files_handler(repo),
                           [chat_repo, chat_file], chat_file, api_visibility="private")
        source.input(catalog_handler, [source, search_query, state], chat_repo,
                     api_visibility="private").success(files_handler, chat_repo, chat_file,
                                                       api_visibility="private")
        account_refresh.click(lambda session: ("My account", catalog_handler("My account", session=session)),
                              state, [source, chat_repo], api_visibility="private").success(
                              files_handler, chat_repo, chat_file, api_visibility="private")
        search_button.click(lambda query, session: ("Search Hub", catalog_handler("Search Hub", query, session)),
                            [search_query, state], [source, chat_repo], api_visibility="private").success(
                            files_handler, chat_repo, chat_file, api_visibility="private")
        search_query.submit(lambda query, session: ("Search Hub", catalog_handler("Search Hub", query, session)),
                            [search_query, state], [source, chat_repo], api_visibility="private").success(
                            files_handler, chat_repo, chat_file, api_visibility="private")
        chat_repo.input(files_handler, chat_repo, chat_file, api_visibility="private")
        chat_file.change(context_limit_handler, [chat_repo, chat_file, state, chat_ctx],
                         [chat_ctx, context_info], api_visibility="private").success(
                         clear_compare, outputs=compare_outputs, api_visibility="private")
        chat_inputs = [state, chat_repo, chat_file, chat_ctx, temperature, new_tokens, top_p,
                       question, without_chat, with_chat]

        def chat_event(*args):
            result = compare_handler(*args)
            return (result[0], gr.update(value=result[1], label=result[5]), *result[2:])

        for event in (send.click, question.submit):
            event(chat_event, chat_inputs, [*compare_outputs, chat_status],
                  api_name="compare" if event == send.click else "compare_submit",
                  concurrency_id="compare", concurrency_limit=1)
        clear.click(clear_compare, outputs=compare_outputs, api_visibility="private")
        refresh.click(refresh_handler, state, review, api_name="status")
        approve.click(lambda session: decision_handler(session, "Approved"), state, [review, archive],
                      api_name="approve", concurrency_id="registry", concurrency_limit=1)
        reject.click(lambda session: decision_handler(session, "Rejected"), state, [review, archive],
                     api_name="reject", concurrency_id="registry", concurrency_limit=1)
        download.click(export_handler, state, archive, api_name="export", concurrency_id="registry", concurrency_limit=1)
    return demo
