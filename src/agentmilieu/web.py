"""Session-scoped Gradio handlers. No credentials or registry ids are client inputs."""

import shutil
import uuid
from functools import lru_cache

import gradio as gr

from . import devices, gpu, llm
from . import interfaces as ui
from .audit import audit_markdown
from .budget import BudgetConfig
from .composer import ComposeOptions
from .devices import DEVICE_PRESETS
from .export import export
from .models import ModelSpec
from .pipeline import BuildOptions, build_pack
from .safety import SafetyPolicy, redact_text
from .store import LocalRegistry
from .tokens import gguf_token_counter


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


def build_handler(session=None, urls="", names="", files=None, purpose="",
                  hf_repo="", hf_file="", device_mode="Probe this machine",
                  device_preset="phone-6gb", ram_gb=6, available_gb=4.5, kv_type="f16",
                  max_ctx=8192, quality_cap=8192, latency_target=0, prefill_tok_s=100,
                  strategy="auto", no_mistral=False, offline=True, scorton=False,
                  model_source="Recommended", model_repo_choices=None, model_file_choices=None):
    try:
        session = dict(session or new_session())
        if offline and (urls.strip() or names.strip() or scorton):
            raise ValueError("Offline mode requires uploaded files; disable URLs/topics and Scorton. "
                             "Hub model headers and tokenizer downloads are still allowed.")
        if not hf_repo or not hf_file:
            raise ValueError("Select a Hub GGUF repository and file in Target")
        model = ui.select_model(hf_repo=hf_repo, hf_file=hf_file)
        counter = gguf_token_counter(hf_repo)
        warnings = []
        if counter is None:
            warnings.append("Warning: token counts are estimated; the with-context prompt may exceed the budget.")
        else:
            warnings.append("Token accounting uses the selected model's Hub tokenizer.json.")
        if device_mode not in ("Probe this machine", "Preset", "Manual"):
            raise ValueError("Unknown device selection mode")
        if device_mode == "Probe this machine":
            device, summary = _probe_device()
            warnings.append(summary)
        else:
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
                            model, device, options, counter=counter)
        session["pack_id"] = result.pack_id
        session["target"] = {"source": model_source, "repo": hf_repo, "file": hf_file,
                             "repos": model_repo_choices or [hf_repo],
                             "files": model_file_choices or [hf_file]}
        pack, registry = result.pack, LocalRegistry(session["root"])
        local = registry.get(result.pack_id)["backend"] == "local"
        budget_text = "\n".join((*pack.budget.explanation, *pack.budget.warnings, *result.warnings, *warnings))
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


def probe_handler():
    try:
        _, summary = _probe_device()
        return redact_text(summary)[0]
    except Exception as exc:
        raise gr.Error(redact_text(str(exc))[0]) from exc


def _probe_device():
    try:
        info = gpu.probe()
    except Exception as exc:  # noqa: BLE001 -- allocation/quota failures can precede worker execution
        info = {"error": f"GPU probe unavailable: {exc}"}
    return devices.probe_machine(info)


def _compare_status(session):
    if not session or not session.get("pack_id"):
        return "Build a pack in this session first. Compare downloads a model from the Hub."
    registry = session_registry(session)
    status = ui.status(session["pack_id"], registry)
    return f"With Milieu context — {status}" + (" (draft; not approved)" if status != "Approved" else "")


def compare_setup(session):
    repos = llm.curated_models()
    context, cap, repo = 8192, 131072, repos[0]
    source, file_update = "Recommended", gr.update(choices=[], value=None)
    ready = bool(session and session.get("pack_id"))
    if ready:
        pack = _session_pack(session)
        context = pack["budget"]["context_window"]
        model = pack["budget"]["model"]
        target = session.get("target")
        if target:
            repo, repos, source = target["repo"], target["repos"], target["source"]
            file_update = gr.update(choices=target["files"], value=target["file"])
        cap = model["n_ctx_train"]
    return (_compare_status(session), gr.update(interactive=ready),
            gr.update(value=context, maximum=max(context, cap)),
            gr.update(choices=repos, value=repo), source,
            file_update, [], [], "", "", "")


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
        preferred = session.get("target", {}).get("repo") if session else None
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
        with_system, without_system = llm.context_prompts(
            pack, question.strip(), counter=gguf_token_counter(repo))
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


def _model_picker():
    with gr.Row():
        source = gr.Radio(["My account", "Recommended", "Search Hub"], value="Recommended", label="Source")
        refresh = gr.Button("Refresh")
        query = gr.Textbox(label="Search Hub")
        search = gr.Button("Search Hub")
    repos = llm.curated_models()
    repo = gr.Dropdown(repos, value=repos[0], label="GGUF repository")
    file = gr.Dropdown([], label="GGUF file (Q4_K_M preferred)")
    return {"source": source, "refresh": refresh, "query": query, "search": search,
            "repo": repo, "file": file, "repos": gr.State(repos), "files": gr.State([])}


def _picker_files(repo):
    update = files_handler(repo)
    return update, update["choices"]


def _wire_model_picker(picker, session, prefix):
    def catalog(source, query, state):
        update = catalog_handler(source, query, state)
        return update, update["choices"]

    def select_source(source, query, state):
        return source, *catalog(source, query, state)

    file_outputs = [picker["file"], picker["files"]]
    picker["source"].input(catalog, [picker["source"], picker["query"], session],
                           [picker["repo"], picker["repos"]], api_name=f"{prefix}_catalog").success(
                           _picker_files, picker["repo"], file_outputs, api_visibility="private")
    picker["refresh"].click(lambda state: select_source("My account", "", state), session,
                            [picker["source"], picker["repo"], picker["repos"]],
                            api_visibility="private").success(
                            _picker_files, picker["repo"], file_outputs, api_visibility="private")
    for event in (picker["search"].click, picker["query"].submit):
        event(lambda query, state: select_source("Search Hub", query, state),
              [picker["query"], session], [picker["source"], picker["repo"], picker["repos"]],
              api_visibility="private").success(
              _picker_files, picker["repo"], file_outputs, api_visibility="private")
    picker["repo"].input(_picker_files, picker["repo"], file_outputs, api_name=f"{prefix}_files")


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
                                  label="Offline mode: uploaded sources; Hub model metadata/tokenizer allowed")
        with gr.Tab("2 · Target"):
            target = _model_picker()
            gr.Markdown("Read the exact GGUF header for model geometry. Tokenizer downloads are best-effort; "
                        "the build warns if token counts must be estimated.")
            device_mode = gr.Radio(["Probe this machine", "Preset", "Manual"],
                                   value="Probe this machine", label="Device selection")
            probe = gr.Button("Probe")
            probe_result = gr.Textbox(label="Detected hardware & budget device", interactive=False, lines=4)
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
            chat = _model_picker()
            chat_repo, chat_file, source = chat["repo"], chat["file"], chat["source"]
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
        _wire_model_picker(target, state, "target")
        _wire_model_picker(chat, state, "compare")
        demo.load(_picker_files, target["repo"], [target["file"], target["files"]], api_visibility="private")
        probe.click(probe_handler, outputs=probe_result, api_name="probe")
        built = build.click(build_handler, [state, urls, names, files, purpose, target["repo"],
            target["file"], device_mode, device, ram, available, kv,
            max_ctx, quality, latency, prefill, strategy, no_mistral, offline, scorton,
            target["source"], target["repos"], target["files"]],
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
