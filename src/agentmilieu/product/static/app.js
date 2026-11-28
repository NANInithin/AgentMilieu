"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("generator");
const field = (name) => form.elements.namedItem(name);
const currency = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 3 });
const money = (mills) => currency.format(mills / 1000);
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let settings;
let selected;
let historyOffset = 0;
let topupAttempt;
let walletData;
let reviewVersion = 0;

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body } = {}) {
  const init = { method, credentials: "same-origin" };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const response = await fetch(path, init);
  if (response.status === 204) return null;
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(result.detail || "The request failed. Try again.", response.status);
  return result;
}

function flash(message = "", failure = false) {
  const node = byId("flash");
  node.textContent = message;
  node.classList.toggle("error", failure);
  node.hidden = !message;
}

function setGuest() {
  byId("account-label").textContent = "Generate cited context packs for your local models.";
  byId("signin").hidden = false;
  byId("account-content").hidden = true;
  byId("logout").hidden = true;
  byId("wallet-toggle").hidden = true;
  field("provider_key").value = "";
  byId("notion-token").value = "";
}

async function action(callback, button) {
  flash();
  if (button) button.disabled = true;
  try {
    return await callback();
  } catch (error) {
    if (error.status === 401) setGuest();
    flash(error.message, true);
  } finally {
    if (button) button.disabled = false;
  }
}

function renderList(id, rows, empty, formatter) {
  const list = byId(id);
  list.replaceChildren();
  for (const row of rows) {
    const item = document.createElement("li");
    const [label, value] = formatter(row);
    item.append(document.createTextNode(label));
    const amount = document.createElement("strong");
    amount.textContent = value;
    item.append(amount);
    list.append(item);
  }
  if (!rows.length) {
    const item = document.createElement("li");
    item.textContent = empty;
    list.append(item);
  }
}

function renderWallet(data) {
  walletData = data;
  byId("balance").textContent = money(data.balance_mills);
  byId("wallet-balance").textContent = money(data.balance_mills);
  renderList("transactions", data.entries, "No transactions yet.", (row) => [row.kind, money(row.amount_mills)]);
  renderList("topups", data.topups, "No top-ups yet.", (row) => [money(row.amount_mills), row.status]);
}

async function refreshWallet(topup) {
  if (topup) return renderWallet(await api(`/api/wallet/topups/${topup}/refresh`, { method: "POST" }));
  renderWallet(await api("/api/wallet"));
}

function renderTopupButtons() {
  for (const euros of settings.topups) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Add €${euros}`;
    button.disabled = !settings.mollie;
    button.addEventListener("click", () => action(async () => {
      if (topupAttempt?.euros !== euros) topupAttempt = { id: crypto.randomUUID(), euros };
      try {
        const result = await api("/api/wallet/topups", { method: "POST", body: topupAttempt });
        window.location.assign(result.url);
      } catch (error) {
        if (error.status === 409) topupAttempt = undefined;
        throw error;
      }
    }, button));
    byId("topup-buttons").append(button);
  }
  if (!settings.mollie) byId("mollie-status").textContent = "Wallet top-ups are currently unavailable.";
}

function highlightPack() {
  for (const button of byId("history").querySelectorAll("button")) {
    button.setAttribute("aria-current", String(button.dataset.pack === selected?.id));
  }
}

async function loadHistory(reset = true) {
  const offset = reset ? 0 : historyOffset;
  const data = await api(`/api/packs?offset=${offset}`);
  const list = byId("history");
  if (reset) list.replaceChildren();
  for (const pack of data.items) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.pack = pack.id;
    button.append(document.createTextNode(pack.metadata.Name));
    const state = document.createElement("small");
    state.textContent = pack.status;
    button.append(state);
    button.addEventListener("click", () => action(() => selectPack(pack.id, true), button));
    li.append(button);
    list.append(li);
  }
  if (offset === 0 && !data.items.length) {
    const li = document.createElement("li");
    li.textContent = "Your generated packs will appear here.";
    list.append(li);
  }
  historyOffset = offset + data.items.length;
  byId("more").hidden = !data.has_more;
  highlightPack();
}

function budgetText(budget) {
  return [`Context window · ${budget.context_window} tokens`,
    ...(budget.explanation || []), ...(budget.warnings || [])].join("\n");
}

async function selectPack(id, focus = false) {
  const version = ++reviewVersion;
  const pack = await api(`/api/packs/${id}`);
  if (version !== reviewVersion) return;
  selected = pack;
  byId("review").hidden = false;
  byId("review-title").textContent = pack.metadata.Name;
  byId("review-status").textContent = pack.status;
  byId("review-meta").textContent = [pack.metadata.Model, pack.metadata.Device, pack.pack.strategy].join(" · ");
  byId("pack-hash").textContent = pack.pack_hash;
  byId("context").textContent = `${pack.pack.system_prompt}\n\n${pack.pack.knowledge}`;
  byId("audit").textContent = pack.audit;
  byId("budget").textContent = [
    budgetText(pack.pack.budget), "", "Token accounting",
    ...Object.entries(pack.pack.token_accounting).map(([key, value]) => `${key} · ${value}`),
  ].join("\n");
  byId("export").hidden = pack.status !== "Approved";
  byId("export").href = `/api/packs/${id}/export`;
  byId("notion-publish").hidden = Boolean(pack.notion_page_id);
  byId("notion-import").hidden = !pack.notion_page_id;
  byId("notion-page").hidden = !pack.notion_page_id;
  if (pack.notion_page_id) byId("notion-page").href = `https://www.notion.so/${encodeURIComponent(pack.notion_page_id.replaceAll("-", ""))}`;
  highlightPack();
  if (focus) byId("review").focus();
}

function updateFields() {
  byId("hub-fields").hidden = field("preset").value !== "hub";
  const kind = field("provider").value;
  byId("provider-fields").hidden = !["openai", "anthropic"].includes(kind);
  field("provider_model").required = !byId("provider-fields").hidden;
  field("provider_key").required = !byId("provider-fields").hidden;
  field("provider_url").placeholder = kind === "anthropic" ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1";
  field("hf_repo").required = !byId("hub-fields").hidden;
  field("hf_file").required = !byId("hub-fields").hidden;
}

function targetOptions() {
  const hub = field("preset").value === "hub";
  return {
    name: field("name").value.trim() || "Memory budget",
    preset: hub ? "" : field("preset").value,
    hf_repo: hub ? field("hf_repo").value.trim() : "",
    hf_file: hub ? field("hf_file").value.trim() : "",
    device_name: field("device_name").value.trim(),
    total_gib: Number(field("total_gib").value),
    available_gib: Number(field("available_gib").value),
    max_context: Number(field("max_context").value),
    kv_type: field("kv_type").value,
  };
}

async function finishGeneration(row) {
  await refreshWallet();
  if (row.status !== "completed") {
    flash(row.error || "Generation failed. Check your sources and wallet.", true);
    return;
  }
  await loadHistory();
  await selectPack(row.pack_id, true);
  flash("Context pack generated. Review it before exporting.");
}

async function trackGeneration(accountId, id, row) {
  const key = `milieu-generation-${accountId}`;
  for (let attempt = 0; attempt < 300; attempt++) {
    if (row && row.status !== "running") {
      sessionStorage.removeItem(key);
      await finishGeneration(row);
      return;
    }
    await pause(3000);
    row = await api(`/api/generations/${id}`);
  }
  flash("Generation is still in progress. Refresh to check its status.");
}

function bindGenerator(accountId) {
  const key = `milieu-generation-${accountId}`;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void action(async () => {
      const pending = sessionStorage.getItem(key);
      if (pending) {
        try {
          await trackGeneration(accountId, pending, await api(`/api/generations/${pending}`));
          return;
        } catch (error) {
          if (error.status !== 404) throw error;
          sessionStorage.removeItem(key);
        }
      }
      if (!field("name").value.trim()) throw new Error("Enter a pack name.");
      const options = {
        ...targetOptions(),
        purpose: field("purpose").value.trim(),
        text: field("text").value,
        urls: field("urls").value.split("\n").map((line) => line.trim()).filter(Boolean),
        strategy: field("strategy").value,
        provider: { kind: field("provider").value, url: field("provider_url").value.trim(),
          model: field("provider_model").value.trim(), key: field("provider_key").value },
      };
      const files = [...byId("files").files];
      if (files.length > 8 || files.reduce((sum, file) => sum + file.size, 0) > 12 * 1024 ** 2) {
        throw new Error("Select up to eight files totalling no more than 12 MiB.");
      }
      const id = crypto.randomUUID();
      const multipart = new FormData();
      multipart.append("id", id);
      multipart.append("options", JSON.stringify(options));
      for (const file of files) multipart.append("files", file);
      sessionStorage.setItem(key, id);
      flash("Generating your context pack…");
      let row;
      try {
        row = await api("/api/generate", { method: "POST", body: multipart });
      } catch (error) {
        if (error.status && error.status < 500) sessionStorage.removeItem(key);
        if (error.status === 402) byId("wallet").hidden = false;
        throw error;
      }
      await trackGeneration(accountId, id, row);
    }, byId("generate"));
  });
  const previous = sessionStorage.getItem(key);
  if (previous) void action(async () => {
    try {
      const row = await api(`/api/generations/${previous}`);
      if (row.status === "running") flash("A generation is still in progress…");
      await trackGeneration(accountId, previous, row);
    } catch (error) {
      if (error.status !== 404) throw error;
      sessionStorage.removeItem(key);
    }
  }, byId("generate"));
}

async function review(decision) {
  const { id, pack_hash } = selected;
  await api(`/api/packs/${id}/review`, { method: "POST", body: { decision, pack_hash } });
  await loadHistory();
  await selectPack(id);
}

async function notion(operation) {
  const id = selected.id;
  const token = byId("notion-token").value.trim();
  if (!token) throw new Error("Enter a Notion integration token.");
  const body = { token };
  if (operation === "publish") body.database_id = byId("notion-db").value.trim();
  const suffix = operation === "publish" ? "" : "/import";
  await api(`/api/packs/${id}/notion${suffix}`, { method: "POST", body });
  await loadHistory();
  await selectPack(id);
  flash(operation === "publish" ? "Pack copied to Notion." : "Notion decision imported.");
}

async function start() {
  settings = await api("/api/config");
  byId("price").textContent = money(settings.price_mills);
  byId("generation-price").textContent = `${money(settings.price_mills)} · failed runs are credited`;
  const me = await api("/api/me").catch((error) => {
    if (error.status !== 401) throw error;
    return null;
  });
  if (!me) {
    setGuest();
    if (!settings.google) {
      byId("google").hidden = true;
      flash("Google sign-in is currently unavailable.");
    }
    return;
  }
  byId("account-content").hidden = false;
  byId("logout").hidden = false;
  byId("wallet-toggle").hidden = false;
  byId("account-label").textContent = me.email;
  byId("balance").textContent = money(me.balance_mills);
  for (const preset of settings.presets) field("preset").add(new Option(preset, preset));
  field("preset").add(new Option("Public Hugging Face GGUF", "hub"));
  field("provider").querySelector('[value="managed"]').disabled = !settings.managed_provider;
  field("preset").addEventListener("change", updateFields);
  field("provider").addEventListener("change", () => {
    if (field("provider").value !== "none") field("strategy").value = "compress";
    updateFields();
  });
  updateFields();
  renderTopupButtons();
  const topup = new URL(window.location.href).searchParams.get("topup");
  byId("wallet").hidden = !topup;
  byId("wallet-toggle").addEventListener("click", () => {
    const wallet = byId("wallet");
    wallet.hidden = !wallet.hidden;
    if (!wallet.hidden) void action(() => refreshWallet(topup));
  });
  byId("wallet-refresh").addEventListener("click", (event) => action(async () => {
    const pending = topup || walletData?.topups.find((item) => ["open", "pending", "authorized"].includes(item.status))?.id;
    await refreshWallet(pending);
  }, event.currentTarget));
  byId("logout").addEventListener("click", (event) => action(async () => {
    await api("/api/logout", { method: "POST" });
    setGuest();
  }, event.currentTarget));
  byId("more").addEventListener("click", (event) => action(() => loadHistory(false), event.currentTarget));
  byId("approve").addEventListener("click", (event) => action(() => review("Approved"), event.currentTarget));
  byId("reject").addEventListener("click", (event) => action(() => review("Rejected"), event.currentTarget));
  byId("notion-publish").addEventListener("click", (event) => action(() => notion("publish"), event.currentTarget));
  byId("notion-import").addEventListener("click", (event) => action(() => notion("import"), event.currentTarget));
  byId("budget-preview").addEventListener("click", (event) => action(async () => {
    byId("budget-result").hidden = true;
    const result = await api("/api/budget", { method: "POST", body: targetOptions() });
    byId("budget-result").textContent = budgetText(result);
    byId("budget-result").hidden = false;
  }, event.currentTarget));
  bindGenerator(me.id);
  await Promise.all([loadHistory(), refreshWallet(topup)]);
  if (topup && walletData?.topups.find((item) => item.id === topup)?.status !== "paid") {
    flash("Your balance updates after Mollie confirms payment.");
  }
}

void action(start);
