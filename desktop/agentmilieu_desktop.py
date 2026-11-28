"""Native Windows client for the AgentMilieu account service."""

import argparse
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
import webbrowser
from contextlib import ExitStack
from pathlib import Path
from tkinter import filedialog, ttk
from urllib.parse import urlsplit
from uuid import uuid4

import requests

# The desktop build adds the compiler's standalone device module to its search path.
from devices import GIB, DeviceProfile, probe_cpu, probe_machine

PRODUCTION = "https://agentmilieu.naninithin.me"
FILE_TYPES = "*.txt *.md *.html *.htm *.csv *.pdf *.docx *.xlsx *.pptx"


class ApiError(ValueError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class Api:
    def __init__(self, server):
        self.server = server.rstrip("/")
        url = urlsplit(self.server)
        if (not url.hostname or url.username or url.path or url.query or url.fragment
                or (url.scheme != "https" and
                    (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1")))):
            raise ValueError("Use an HTTPS AgentMilieu origin.")
        self.token = ""

    def request(self, path, method="GET", *, timeout=25, raw=False, **kwargs):
        try:
            headers = {"Authorization": "Bearer " + self.token} if self.token else {}
            response = requests.request(method, self.server + path, headers=headers,
                                        timeout=(5, timeout), allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raise ApiError("Could not reach AgentMilieu. Check your connection and try again.", 0) from exc
        if response.status_code == 204:
            return None
        if not 200 <= response.status_code < 300:
            try:
                message = response.json().get("detail")
            except ValueError:
                message = None
            raise ApiError(message if isinstance(message, str) else "The AgentMilieu request failed.",
                           response.status_code)
        return response.content if raw else response.json()


def stored_token(server, token=None):
    try:
        import pywintypes
        import win32cred
    except ImportError:
        return ""
    target = "AgentMilieu account " + server
    kind = win32cred.CRED_TYPE_GENERIC
    try:
        if token is None:
            return win32cred.CredRead(target, kind, 0)["CredentialBlob"].decode("utf-8")
        if token:
            win32cred.CredWrite({"Type": kind, "TargetName": target, "UserName": "AgentMilieu",
                                 "CredentialBlob": token.encode(), "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE}, 0)
        else:
            win32cred.CredDelete(target, kind, 0)
    except (pywintypes.error, UnicodeError, AttributeError, KeyError):
        return ""
    return token


def money(mills):
    sign = "-" if mills < 0 else ""
    euros, fraction = divmod(abs(mills), 1000)
    return f"{sign}€{euros}.{fraction // 10:02d}" if fraction % 10 == 0 else f"{sign}€{euros}.{fraction:03d}"


def budget_lines(budget):
    return "\n".join([f"Context window: {budget['context_window']} tokens",
                      *budget["explanation"], *budget["warnings"]])


def local_probe():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="replace", capture_output=True, check=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, free = result.stdout.splitlines()[0].rsplit(",", 2)
        gpu = DeviceProfile(int(total) * 1024 ** 2, int(free) * 1024 ** 2, name.strip())
        return probe_machine({"name": gpu.name, "total_ram": gpu.total_ram,
                              "available_ram": gpu.available_ram})
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        cpu = probe_cpu()
        return cpu, f"CPU RAM: {cpu.available_ram / GIB:.2f} GiB available of {cpu.total_ram / GIB:.2f} GiB"


def text_set(widget, content):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


class Desktop:
    def __init__(self, window, api, *, connect=True):
        self.window, self.api = window, api
        self.callbacks = queue.Queue()
        self.account = None
        self.packs = []
        self.pack = None
        self.files = ()
        self.generation_id = None
        self.checkout_attempt = None
        self.login_version = 0
        self.generate_busy = False
        self.settings = {"price_mills": 499, "topups": [5, 10, 20], "managed_provider": False, "mollie": False}
        self.vars = {}
        self.status = tk.StringVar(value="Set up your sources and device. Sign in when you’re ready to generate.")
        self.identity = tk.StringVar(value="Guest")
        self.balance = tk.StringVar(value="Wallet unavailable until sign-in")
        self.probe_status = tk.StringVar()
        self.build_window()
        self.window.after(80, self.drain)
        if connect:
            self.api.token = stored_token(self.api.server)
            self.work(self.load_config, self.set_config)
            if self.api.token:
                self.work(self.load_account, self.set_account)

    def build_window(self):
        self.window.title("AgentMilieu")
        self.window.geometry("960x820")
        self.window.minsize(680, 610)
        outer = ttk.Frame(self.window, padding=18)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 13))
        ttk.Label(header, text="AgentMilieu.", foreground="#49248e", font=("Segoe UI", 25, "bold")).pack(side="left")
        self.signin_button = ttk.Button(header, text="Continue with Google", command=self.signin)
        self.signin_button.pack(side="right")
        ttk.Label(header, textvariable=self.identity).pack(side="right", padx=12)

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        setup = ttk.Frame(self.tabs, padding=12)
        self.review_tab = ttk.Frame(self.tabs, padding=12)
        wallet = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(setup, text="Build")
        self.tabs.add(self.review_tab, text="Review")
        self.tabs.add(wallet, text="Wallet")
        canvas = tk.Canvas(setup, highlightthickness=0)
        scroll = ttk.Scrollbar(setup, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(fill="both", expand=True)
        client = ttk.Frame(canvas, padding=(0, 0, 12, 6))
        view = canvas.create_window((0, 0), window=client, anchor="nw")
        client.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(view, width=event.width))
        setup.bind_all("<MouseWheel>", lambda event: self.scroll_setup(event, canvas, setup), add="+")
        self.build_form(client)
        self.build_review(self.review_tab)
        self.build_wallet(wallet)
        ttk.Label(outer, textvariable=self.status, wraplength=650).pack(anchor="w", pady=(12, 0))

    def scroll_setup(self, event, canvas, setup):
        if self.tabs.select() == str(setup) and not isinstance(event.widget, (tk.Text, ttk.Combobox)):
            canvas.yview_scroll(-int(event.delta / 120), "units")

    def field(self, parent, label, key, initial="", choices=None, secret=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=26).pack(side="left")
        variable = self.vars[key] = tk.StringVar(value=initial)
        if choices is None:
            widget = ttk.Entry(row, textvariable=variable, show="•" if secret else "")
        else:
            widget = ttk.Combobox(row, textvariable=variable, values=choices, state="readonly")
        widget.pack(fill="x", expand=True)
        return widget

    def build_form(self, parent):
        ttk.Label(parent, text="New context pack", font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(0, 7))
        self.field(parent, "Pack name", "name")
        self.field(parent, "Purpose", "purpose")
        sources = ttk.LabelFrame(parent, text="Sources", padding=10)
        sources.pack(fill="x", pady=8)
        buttons = ttk.Frame(sources)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Choose files", command=self.select_files).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear_files).pack(side="left", padx=5)
        self.file_label = ttk.Label(buttons, text="No files selected", wraplength=420)
        self.file_label.pack(side="left", padx=5)
        ttk.Label(sources, text="Upload source files to AgentMilieu for processing. Up to 8 files, 12 MiB total.").pack(anchor="w", pady=(6, 0))
        ttk.Label(sources, text="Paste text").pack(anchor="w", pady=(7, 0))
        self.pasted = tk.Text(sources, height=4, wrap="word", font=("Segoe UI", 10))
        self.pasted.pack(fill="x")
        ttk.Label(sources, text="Public HTTPS links, one per line").pack(anchor="w", pady=(7, 0))
        self.urls = tk.Text(sources, height=2, wrap="word", font=("Segoe UI", 10))
        self.urls.pack(fill="x")

        target = ttk.LabelFrame(parent, text="Target model and device", padding=10)
        target.pack(fill="x", pady=8)
        self.models = self.field(target, "Model", "preset", "Llama-3.2-1B",
                                 ["Llama-3.2-1B", "Public Hugging Face GGUF"])
        self.models.bind("<<ComboboxSelected>>", lambda _event: self.toggle_fields())
        self.hf_repo = self.field(target, "Hugging Face repository", "hf_repo")
        self.hf_file = self.field(target, "GGUF filename", "hf_file")
        self.field(target, "Device label", "device_name", "Manual device")
        self.field(target, "Total memory (GiB)", "total_gib", "8")
        self.field(target, "Available memory (GiB)", "available_gib", "6")
        self.field(target, "Maximum context", "max_context", "8192")
        self.field(target, "KV cache", "kv_type", "f16", ("f16", "q8_0", "q4_0"))
        tools = ttk.Frame(target)
        tools.pack(fill="x", pady=5)
        ttk.Button(tools, text="Probe this machine", command=lambda: self.work(local_probe, self.use_probe)).pack(side="left")
        ttk.Button(tools, text="Estimate memory budget", command=self.estimate).pack(side="left", padx=6)
        ttk.Label(target, textvariable=self.probe_status, wraplength=610).pack(anchor="w")

        composition = ttk.LabelFrame(parent, text="Composition", padding=10)
        composition.pack(fill="x", pady=8)
        self.field(composition, "Strategy", "strategy", "auto", ("auto", "stuff", "compress", "rag"))
        self.provider = self.field(composition, "Compression provider", "provider", "Extractive",
                                   ("Extractive", "OpenAI-compatible", "Anthropic-compatible"))
        self.provider.bind("<<ComboboxSelected>>", self.change_provider)
        self.provider_url = self.field(composition, "API base URL (optional)", "provider_url")
        self.provider_model = self.field(composition, "API model", "provider_model")
        self.provider_key = self.field(composition, "API key", "provider_key", secret=True)
        ttk.Label(composition, text="Model-assisted compression sends selected redacted chunks to the provider.",
                  wraplength=620).pack(anchor="w", pady=(5, 0))
        self.generate = ttk.Button(parent, text="Generate context pack · €0.499", command=self.submit, state="disabled")
        self.generate.pack(anchor="w", pady=6)
        self.toggle_fields()

    def build_review(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 8))
        self.history = ttk.Combobox(toolbar, state="readonly")
        self.history.pack(side="left", fill="x", expand=True)
        self.history.bind("<<ComboboxSelected>>", self.select_pack)
        ttk.Button(toolbar, text="Refresh packs", command=self.refresh_history).pack(side="left", padx=6)
        self.pack_label = ttk.Label(parent, text="Generate a pack to start a review.", wraplength=600)
        self.pack_label.pack(anchor="w", pady=(0, 7))
        panes = ttk.Notebook(parent)
        panes.pack(fill="both", expand=True)
        self.readers = {}
        for key, title in (("context", "Compiled context"), ("audit", "Audit"), ("budget", "Budget")):
            frame = ttk.Frame(panes)
            panes.add(frame, text=title)
            reader = tk.Text(frame, wrap="word", state="disabled", font=("Consolas", 10))
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=reader.yview)
            reader.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            reader.pack(fill="both", expand=True)
            self.readers[key] = reader
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=8)
        self.review_buttons = []
        for title, command in (("Approve", lambda: self.decide("Approved")),
                               ("Reject", lambda: self.decide("Rejected")),
                               ("Download ZIP", self.download),
                               ("Notion", self.notion)):
            button = ttk.Button(actions, text=title, command=command, state="disabled")
            button.pack(side="left", padx=(0, 6))
            self.review_buttons.append(button)

    def build_wallet(self, parent):
        ttk.Label(parent, text="Wallet", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(parent, textvariable=self.balance, font=("Segoe UI", 12)).pack(anchor="w", pady=7)
        actions = ttk.Frame(parent)
        actions.pack(anchor="w", pady=6)
        self.topup_buttons = []
        for euros in (5, 10, 20):
            button = ttk.Button(actions, text=f"Add €{euros}", command=lambda value=euros: self.topup(value),
                                state="disabled")
            button.pack(side="left", padx=(0, 6))
            self.topup_buttons.append(button)
        ttk.Button(actions, text="Refresh balance", command=lambda: self.refresh_wallet(True)).pack(side="left")
        ttk.Label(parent, text="Mollie checkout opens in your browser. Refresh after paying.").pack(anchor="w", pady=(0, 8))
        self.activity = tk.Text(parent, height=12, wrap="word", state="disabled", font=("Consolas", 10))
        self.activity.pack(fill="both", expand=True)

    def change_provider(self, _event):
        if self.vars["provider"].get() != "Extractive":
            self.vars["strategy"].set("compress")
        self.toggle_fields()

    def toggle_fields(self):
        hub = self.vars["preset"].get() == "Public Hugging Face GGUF"
        custom = self.vars["provider"].get() in ("OpenAI-compatible", "Anthropic-compatible")
        for widget, enabled in ((self.hf_repo, hub), (self.hf_file, hub), (self.provider_url, custom),
                                (self.provider_model, custom), (self.provider_key, custom)):
            widget.configure(state="normal" if enabled else "disabled")

    def work(self, operation, ready, failed=None):
        def run():
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001 -- report failures back on the UI thread
                self.callbacks.put((failed or self.show_error, exc))
            else:
                self.callbacks.put((ready, result))
        threading.Thread(target=run, daemon=True).start()

    def drain(self):
        while not self.callbacks.empty():
            callback, data = self.callbacks.get_nowait()
            callback(data)
        self.window.after(80, self.drain)

    def show_error(self, error):
        if isinstance(error, ApiError) and error.status == 401:
            self.clear_session()
        self.status.set(str(error))

    def load_config(self):
        return self.api.request("/api/config")

    def set_config(self, data):
        self.settings = data
        self.models["values"] = [*data["presets"], "Public Hugging Face GGUF"]
        values = ["Extractive", "OpenAI-compatible", "Anthropic-compatible"]
        if data["managed_provider"]:
            values.insert(1, "AgentMilieu managed")
        self.provider["values"] = values
        self.refresh_buttons()

    def refresh_buttons(self):
        self.generate.configure(state="normal" if self.account and not self.generate_busy else "disabled")
        for button in self.topup_buttons:
            button.configure(state="normal" if self.account and self.settings["mollie"] else "disabled")

    def load_account(self):
        return self.api.request("/api/me")

    def set_account(self, account):
        self.account = account
        self.identity.set(account["email"])
        self.signin_button.configure(text="Sign out", command=self.signout, state="normal")
        self.status.set("Signed in. Your wallet and packs are synced with the web workspace.")
        self.refresh_buttons()
        self.refresh_wallet()
        self.refresh_history()

    def clear_session(self):
        self.account = self.pack = None
        self.checkout_attempt = self.generation_id = None
        self.api.token = ""
        stored_token(self.api.server, "")
        self.identity.set("Guest")
        self.balance.set("Wallet unavailable until sign-in")
        self.signin_button.configure(text="Continue with Google", command=self.signin, state="normal")
        self.history.set("")
        self.history["values"] = ()
        self.pack_label.configure(text="Generate a pack to start a review.")
        self.vars["provider_key"].set("")
        for widget in (*self.readers.values(), self.activity):
            text_set(widget, "")
        for button in self.review_buttons:
            button.configure(state="disabled")
        self.refresh_buttons()

    def signin(self):
        self.login_version += 1
        version = self.login_version
        self.signin_button.configure(state="disabled")
        self.work(lambda: self.api.request("/api/auth/desktop/start", "POST"),
                  lambda flow: self.open_login(flow, version), self.login_failed)

    def login_failed(self, error):
        self.signin_button.configure(text="Continue with Google", command=self.signin, state="normal")
        self.show_error(error)

    def cancel_login(self):
        self.login_version += 1
        self.signin_button.configure(text="Continue with Google", command=self.signin, state="normal")
        self.status.set("Sign-in canceled.")

    def open_login(self, flow, version):
        self.status.set(f"Enter code {flow['code']} in your browser, then return here.")
        self.signin_button.configure(text=f"Cancel sign-in  ·  {flow['code']}", command=self.cancel_login, state="normal")
        webbrowser.open(flow["url"])

        def poll():
            deadline = time.monotonic() + 590
            while time.monotonic() < deadline and version == self.login_version:
                time.sleep(3)
                response = self.api.request("/api/auth/desktop/exchange", "POST",
                                            json={"id": flow["id"], "secret": flow["secret"]})
                if "token" in response:
                    return response["token"]
            raise ValueError("Windows sign-in expired. Try again.")

        def complete(token):
            if version == self.login_version:
                self.api.token = token
                stored_token(self.api.server, token)
                self.work(self.load_account, self.set_account, self.login_failed)

        def failed(error):
            if version == self.login_version:
                self.login_failed(error)

        self.work(poll, complete, failed)

    def signout(self):
        self.signin_button.configure(state="disabled")
        self.work(lambda: self.api.request("/api/logout", "POST"), lambda _data: self.clear_session(),
                  lambda error: (self.signin_button.configure(state="normal"), self.show_error(error)))

    def select_files(self):
        selected = filedialog.askopenfilenames(title="Select context sources",
                                               filetypes=[("Supported documents", FILE_TYPES)])
        if selected:
            self.files = tuple(Path(item) for item in selected)
            self.file_label.configure(text=", ".join(item.name for item in self.files))

    def clear_files(self):
        self.files = ()
        self.file_label.configure(text="No files selected")

    def use_probe(self, result):
        device, summary = result
        self.vars["total_gib"].set(f"{device.total_ram / GIB:.6f}")
        self.vars["available_gib"].set(f"{device.available_ram / GIB:.6f}")
        self.vars["device_name"].set(device.name[:120])
        self.probe_status.set(summary)
        self.status.set("Machine probe complete.")

    def target(self):
        values = {key: variable.get().strip() for key, variable in self.vars.items()}
        hub = values["preset"] == "Public Hugging Face GGUF"
        try:
            total, available = float(values["total_gib"]), float(values["available_gib"])
            context = int(values["max_context"])
            if not 0 < available <= total <= 65536:
                raise ValueError("Invalid memory")
        except ValueError as exc:
            raise ValueError("Check total and available memory and the maximum context.") from exc
        return {"name": values["name"] or "Memory budget", "preset": "" if hub else values["preset"],
                "hf_repo": values["hf_repo"] if hub else "", "hf_file": values["hf_file"] if hub else "",
                "device_name": values["device_name"], "total_gib": total, "available_gib": available,
                "max_context": context, "kv_type": values["kv_type"]}

    def estimate(self):
        if not self.account:
            self.status.set("Sign in to estimate a model budget. Machine probing is available in guest mode.")
            return
        try:
            target = self.target()
        except ValueError as exc:
            self.show_error(exc)
            return
        self.work(lambda: self.api.request("/api/budget", "POST", json=target),
                  lambda budget: self.show_budget(budget_lines(budget)))

    def show_budget(self, content):
        dialog = tk.Toplevel(self.window)
        dialog.title("Memory budget")
        dialog.geometry("670x340")
        reader = tk.Text(dialog, wrap="word", padx=12, pady=12)
        reader.pack(fill="both", expand=True)
        text_set(reader, content)

    def submit(self):
        if self.generate_busy or not self.account:
            return
        try:
            options = self.target()
            if not self.vars["name"].get().strip():
                raise ValueError("Enter a pack name.")
            provider = {"Extractive": "none", "AgentMilieu managed": "managed",
                        "OpenAI-compatible": "openai", "Anthropic-compatible": "anthropic"}[self.vars["provider"].get()]
            custom = provider in ("openai", "anthropic")
            options.update(purpose=self.vars["purpose"].get().strip(), strategy=self.vars["strategy"].get(),
                           text=self.pasted.get("1.0", "end-1c"), urls=self.urls.get("1.0", "end-1c").split(),
                           provider={"kind": provider, "url": self.vars["provider_url"].get().strip() if custom else "",
                                     "model": self.vars["provider_model"].get().strip() if custom else "",
                                     "key": self.vars["provider_key"].get().strip() if custom else ""})
            files = self.files
            if len(files) > 8 or sum(file.stat().st_size for file in files) > 12 * 1024 ** 2:
                raise ValueError("Select up to eight files totalling no more than 12 MiB.")
        except (ValueError, OSError) as exc:
            self.show_error(exc)
            return
        self.generate_busy = True
        self.refresh_buttons()
        self.status.set("Generating your context pack…")
        previous = self.generation_id
        generation_id = previous or str(uuid4())
        self.generation_id = generation_id

        def compile_or_resume():
            row = None
            path = f"/api/generations/{generation_id}"
            if previous:
                try:
                    row = self.api.request(path)
                except ApiError as exc:
                    if exc.status != 404:
                        raise
            if row is None:
                # Even without uploads, send an explicitly multipart request.
                with ExitStack() as stack:
                    parts = [("id", (None, generation_id)), ("options", (None, json.dumps(options)))]
                    parts.extend(("files", (file.name, stack.enter_context(file.open("rb")))) for file in files)
                    row = self.api.request("/api/generate", "POST", timeout=180, files=parts)
            while row["status"] == "running":
                time.sleep(3)
                row = self.api.request(path)
            return row

        self.work(compile_or_resume, self.generation_finished, self.generation_failed)

    def generation_finished(self, result):
        self.generation_id = None
        self.generate_busy = False
        self.refresh_buttons()
        self.refresh_wallet()
        if result["status"] != "completed":
            self.status.set(result["error"] or "Generation failed. Check your sources and the wallet.")
            return
        self.status.set("Pack generated. Review it before exporting.")
        self.refresh_history(result["pack_id"])
        self.tabs.select(self.review_tab)

    def generation_failed(self, error):
        self.generate_busy = False
        if isinstance(error, ApiError):
            if 400 <= error.status < 500:
                self.generation_id = None
            if error.status == 402:
                self.tabs.select(2)
        self.refresh_buttons()
        self.show_error(error)

    def refresh_history(self, select=None):
        if not self.account:
            return
        self.work(lambda: self.api.request("/api/packs")["items"],
                  lambda rows: self.render_history(rows, select))

    def render_history(self, rows, select):
        self.packs = rows
        self.history["values"] = [f"{item['metadata']['Name']}   ·   {item['status']}" for item in rows]
        if rows:
            index = next((i for i, item in enumerate(rows) if item["id"] == select), 0)
            self.history.current(index)
            self.select_pack()

    def select_pack(self, _event=None):
        index = self.history.current()
        if index >= 0:
            self.pack = None
            for button in self.review_buttons:
                button.configure(state="disabled")
            pack_id = self.packs[index]["id"]
            self.work(lambda: self.api.request(f"/api/packs/{pack_id}"), self.render_pack)

    def render_pack(self, pack):
        index = self.history.current()
        if not self.account or index < 0 or self.packs[index]["id"] != pack["id"]:
            return
        self.pack = pack
        context = pack["pack"]
        self.pack_label.configure(text=f"{pack['metadata']['Name']} · {pack['status']} · {pack['pack_hash']}")
        text_set(self.readers["context"], context["system_prompt"] + "\n\n" + context["knowledge"])
        text_set(self.readers["audit"], pack["audit"])
        accounting = "\n".join(f"{name}: {value}" for name, value in context["token_accounting"].items())
        text_set(self.readers["budget"], budget_lines(context["budget"]) + "\n\nToken accounting\n" + accounting)
        for index, button in enumerate(self.review_buttons):
            button.configure(state="disabled" if index == 2 and pack["status"] != "Approved" else "normal")

    def decide(self, decision):
        pack_id, digest = self.pack["id"], self.pack["pack_hash"]
        self.work(lambda: self.api.request(f"/api/packs/{pack_id}/review", "POST",
                                          json={"decision": decision, "pack_hash": digest}),
                  lambda _data: self.refresh_history(pack_id))

    def download(self):
        pack_id = self.pack["id"]
        destination = filedialog.asksaveasfilename(defaultextension=".zip", initialfile=f"AgentMilieu-{pack_id}.zip",
                                                   filetypes=[("ZIP archive", "*.zip")])
        if destination:
            def save():
                raw = self.api.request(f"/api/packs/{pack_id}/export", raw=True)
                Path(destination).write_bytes(raw)
            self.work(save, lambda _data: self.status.set("Approved pack downloaded."))

    def notion(self):
        pack = self.pack
        dialog = tk.Toplevel(self.window)
        dialog.title("Notion review")
        panel = ttk.Frame(dialog, padding=15)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Notion integration token").pack(anchor="w")
        token = ttk.Entry(panel, width=40, show="•")
        token.pack(fill="x")
        database = ttk.Entry(panel, width=40)
        linked = pack["notion_page_id"]
        if linked:
            ttk.Button(panel, text="Open linked page", command=lambda: webbrowser.open(
                "https://www.notion.so/" + linked.replace("-", ""))).pack(anchor="w", pady=6)
        else:
            ttk.Label(panel, text="Notion database ID").pack(anchor="w", pady=(5, 0))
            database.pack(fill="x")

        def send():
            if not token.get().strip():
                self.status.set("Enter a Notion integration token.")
                return
            body = {"token": token.get().strip()}
            if not linked:
                body["database_id"] = database.get().strip()
            dialog.destroy()
            suffix = "/import" if linked else ""
            self.work(lambda: self.api.request(f"/api/packs/{pack['id']}/notion{suffix}", "POST", json=body),
                      lambda _data: self.refresh_history(pack["id"]))

        ttk.Button(panel, text="Import decision" if linked else "Send to Notion", command=send).pack(anchor="w", pady=6)
        token.focus_set()

    def refresh_wallet(self, verify=False):
        if not self.account:
            self.status.set("Sign in to load your wallet.")
            return

        def load():
            data = self.api.request("/api/wallet")
            if verify:
                pending = [item for item in data["topups"] if item["status"] in ("open", "pending", "authorized")]
                for item in pending[:3]:
                    data = self.api.request(f"/api/wallet/topups/{item['id']}/refresh", "POST")
            return data
        self.work(load, self.render_wallet)

    def render_wallet(self, data):
        if not self.account:
            return
        self.balance.set(f"Balance  {money(data['balance_mills'])}")
        entries = [f"{money(row['amount_mills']):>10}  {row['kind']}" for row in data["entries"]]
        topups = [f"{money(row['amount_mills']):>10}  {row['status']}" for row in data["topups"]]
        text_set(self.activity, "\n".join(["Recent activity", *(entries or ["No activity yet."]),
                                         "", "Top-ups", *(topups or ["No top-ups yet."])]))

    def topup(self, euros):
        if not self.checkout_attempt or self.checkout_attempt["euros"] != euros:
            self.checkout_attempt = {"id": str(uuid4()), "euros": euros}
        attempt = self.checkout_attempt

        def open_checkout(data):
            self.checkout_attempt = None
            webbrowser.open(data["url"])
            self.status.set("Mollie checkout opened. Return here and refresh your balance after paying.")

        def failed(error):
            if isinstance(error, ApiError) and error.status == 409:
                self.checkout_attempt = None
            self.show_error(error)

        self.work(lambda: self.api.request("/api/wallet/topups", "POST", json=attempt), open_checkout, failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=PRODUCTION)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    window = tk.Tk()
    Desktop(window, Api(args.server), connect=not args.smoke_test)
    if args.smoke_test:
        window.withdraw()
        window.update_idletasks()
        local_probe()
        window.destroy()
        return
    window.mainloop()


if __name__ == "__main__":
    main()
