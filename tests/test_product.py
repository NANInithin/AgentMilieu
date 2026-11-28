import concurrent.futures
import io
import json
import os
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from zipfile import ZipFile

import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("google.auth")
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from agentmilieu.product import auth, outbound
from agentmilieu.product.config import COOKIE, PRICE_MILLS, Settings
from agentmilieu.product.db import Conflict, Database
from agentmilieu.product.payments import Mollie
from agentmilieu.product.server import create_app
from agentmilieu.store import NotionStore


@pytest.fixture
def setup(tmp_path):
    url = os.getenv("PRODUCT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set PRODUCT_TEST_DATABASE_URL to run the PostgreSQL product tests.")
    schema = "test_milieu_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    installer = tmp_path / "AgentMilieu-Setup.exe"
    installer.write_bytes(b"test response only")
    settings = Settings(make_conninfo(url, options=f"-csearch_path={schema}"), "https://testserver",
                        "temporary-test-session-secret-12345", "google-id", "google-secret",
                        "test_mock", managed_model="managed-test-model",
                        managed_key="managed-test-secret", installer=installer)
    try:
        db = Database(settings.database_url)
        db.initialize()
        with TestClient(create_app(settings), base_url=settings.public_url) as client:
            yield db, client, settings
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def account(db, email="owner@example.com"):
    return db.account({"sub": uuid4().hex, "email": email, "name": "Test user"})


def credit(db, owner, amount=5000):
    ref = uuid4()
    db.start_topup(owner, ref, amount)
    assert db.apply_payment(ref, "tr_" + uuid4().hex, amount, "paid", None, 0)
    return ref


def test_oauth_desktop_exchange_and_revocation(setup, monkeypatch):
    _db, client, settings = setup
    start = client.post("/api/auth/desktop/start").json()
    link = client.get(start["url"], follow_redirects=False).headers["location"]
    assert link.startswith("/auth/google?desktop=")
    redirect = client.get(link, follow_redirects=False)
    query = parse_qs(urlsplit(redirect.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert client.get("/auth/google/callback?code=test&state=wrong").status_code == 400
    redirect = client.get(link, follow_redirects=False)
    query = parse_qs(urlsplit(redirect.headers["location"]).query)

    class FakeToken:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id_token": "mock"}

    def verify(token, request, audience):
        assert token == "mock" and audience == settings.google_id
        return {"sub": "stable-subject", "email": "windows@example.com",
                "email_verified": True, "nonce": query["nonce"][0]}

    monkeypatch.setattr(auth.requests, "post", lambda *args, **kwargs: FakeToken())
    monkeypatch.setattr(auth.id_token, "verify_oauth2_token", verify)
    response = client.get("/auth/google/callback",
                          params={"code": "test", "state": query["state"][0]},
                          follow_redirects=False)
    assert response.status_code == 303
    assert "HttpOnly" in response.headers.get_list("set-cookie")[0]
    assert "Secure" in response.headers.get_list("set-cookie")[0]
    assert response.headers["referrer-policy"] == "no-referrer"
    confirmation = client.get(response.headers["location"])
    assert confirmation.headers["referrer-policy"] == "same-origin"
    csrf = confirmation.text.split('name="csrf" value="')[1].split('"')[0]
    exchange = {"id": start["id"], "secret": start["secret"]}
    assert client.post("/api/auth/desktop/exchange", json=exchange).status_code == 202
    form = {"code": start["code"], "csrf": csrf}
    assert client.post(response.headers["location"], data=form).status_code == 403
    for origin in ("null", "https://other.example"):
        assert client.post(response.headers["location"], data=form,
                           headers={"Origin": origin}).status_code == 403
    approved = client.post(response.headers["location"], data=form, headers={"Origin": settings.public_url})
    assert approved.status_code == 200
    token = client.post("/api/auth/desktop/exchange", json=exchange).json()["token"]
    assert client.post("/api/auth/desktop/exchange", json=exchange).status_code == 202
    client.cookies.clear()
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/me", headers=headers).json()["email"] == "windows@example.com"
    assert client.post("/api/logout", headers=headers).status_code == 204
    assert client.get("/api/me", headers=headers).status_code == 401


def test_checkout_idempotency_webhook_and_reversal(setup, monkeypatch):
    db, client, settings = setup
    owner = account(db)
    client.cookies.set(COOKIE, db.new_session(owner))
    headers = {"Origin": settings.public_url}
    topup_id = str(uuid4())
    calls = []
    payment = {"id": "tr_test", "status": "open", "amount": {"currency": "EUR", "value": "5.00"},
               "metadata": {"agentmilieu_topup": topup_id},
               "_links": {"checkout": {"href": "https://checkout.mollie.com/test"}}}

    def request(_self, method, path, *, payload=None, key=None):
        calls.append((method, path, payload, key))
        return payment

    monkeypatch.setattr(Mollie, "_request", request)
    topup = {"id": topup_id, "euros": 5}
    assert client.post("/api/wallet/topups", json=topup).status_code == 403
    for _ in range(2):
        assert client.post("/api/wallet/topups", json=topup, headers=headers).json()["url"].startswith("https://")
    assert len(calls) == 1 and str(calls[0][3]) == topup_id
    assert calls[0][2]["amount"]["value"] == "5.00"
    assert calls[0][2]["webhookUrl"] == settings.public_url + f"/api/mollie/webhook/{topup_id}"
    assert client.get("/api/wallet").json()["balance_mills"] == 0
    payment.update(status="paid")
    for _ in range(3):
        assert client.post(f"/api/mollie/webhook/{topup_id}", data={"id": "tr_test"}).status_code == 200
    assert client.get("/api/wallet").json()["balance_mills"] == 5000
    payment["amountRefunded"] = {"currency": "EUR", "value": "2.00"}
    for _ in range(2):
        assert client.post(f"/api/mollie/webhook/{topup_id}", data={"id": "tr_test"}).status_code == 200
    wallet = client.get("/api/wallet").json()
    assert wallet["balance_mills"] == 3000
    assert wallet["topups"][0]["status"] == "partially reversed"
    assert len(wallet["entries"]) == 2
    assert client.post(f"/api/mollie/webhook/{uuid4()}", data={"id": "tr_test"}).status_code == 200
    updated = dict(payment, amountRefunded=None)
    payment.update(updated)
    assert client.post(f"/api/mollie/webhook/{topup_id}", data={"id": "tr_test"}).status_code == 200
    assert client.get("/api/wallet").json()["balance_mills"] == 5000


def test_paid_generation_and_hash_gated_export(setup):
    db, client, _settings = setup
    owner = account(db)
    headers = {"Authorization": "Bearer " + db.new_session(owner)}
    assert client.get("/").status_code == 200
    assert client.get("/app").status_code == 200
    assert client.head("/downloads/AgentMilieu-Setup.exe").status_code == 200
    assert "attachment" in client.get("/downloads/AgentMilieu-Setup.exe").headers["content-disposition"]
    assert client.get("/api/packs").status_code == 401
    fields = {"name": "Release guide", "total_gib": 8, "available_gib": 6,
              "text": "The release guide describes packaging and review steps."}
    assert client.post("/api/budget", json=fields, headers=headers).json()["context_window"] == 8192
    expanded = client.post("/api/budget", json={**fields, "max_context": 16384}, headers=headers)
    assert expanded.json()["context_window"] == 16384
    request_id = str(uuid4())

    def generate(options, generation_id=request_id):
        parts = [("id", (None, generation_id)), ("options", (None, json.dumps(options)))]
        return client.post("/api/generate", headers=headers, files=parts)

    assert generate(fields).status_code == 402
    credit(db, owner)
    result = generate(fields).json()
    assert result["status"] == "completed"
    assert generate(fields).json() == result
    pack_id = result["pack_id"]
    assert client.get(f"/api/generations/{request_id}", headers=headers).json() == result
    detail = client.get(f"/api/packs/{pack_id}", headers=headers).json()
    assert detail["status"] == "Pending review"
    assert "packaging and review steps" in detail["pack"]["knowledge"]
    export = f"/api/packs/{pack_id}/export"
    assert client.get(export, headers=headers).status_code == 409
    review = f"/api/packs/{pack_id}/review"
    assert client.post(review, headers=headers, json={"decision": "Approved", "pack_hash": "0" * 64}).status_code == 409
    assert client.post(review, headers=headers,
                       json={"decision": "Approved", "pack_hash": detail["pack_hash"]}).status_code == 200
    with ZipFile(io.BytesIO(client.get(export, headers=headers).content)) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert manifest["pack_hash"] == detail["pack_hash"]
        assert "audit.md" in pack.namelist()
    other = {"Authorization": "Bearer " + db.new_session(account(db, "other@example.com"))}
    assert client.get(export, headers=other).status_code == 404
    assert client.get("/api/packs", headers=other).json()["items"] == []
    failed = generate({**fields, "text": " ", "urls": ["https://127.0.0.1/data"]}, str(uuid4())).json()
    assert failed["status"] == "failed"
    assert client.get("/api/wallet", headers=headers).json()["balance_mills"] == 5000 - PRICE_MILLS
    tampered = detail["pack_hash"]
    with db.connect() as conn:
        conn.execute("UPDATE packs SET payload = jsonb_set(payload, '{files,context.md}', '\"tampered\"') WHERE id = %s",
                     (pack_id,))
    assert client.get(export, headers=headers).status_code == 409
    assert client.post(review, headers=headers,
                       json={"decision": "Approved", "pack_hash": tampered}).status_code == 409


def test_only_one_reservation_and_idempotent_recovery(setup):
    db, _client, _settings = setup
    owner = account(db)
    credit(db, owner)
    jobs = [uuid4(), uuid4()]

    def try_reserve(job):
        try:
            db.reserve_generation(owner, job)
            return job
        except Conflict:
            return None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        active = [job for job in pool.map(try_reserve, jobs) if job is not None]
    assert len(active) == 1
    with db.connect() as conn:
        conn.execute("UPDATE generations SET lease_until = now() WHERE id = %s", (active[0],))
    db.reap()
    db.reap()
    assert db.wallet(owner)["balance_mills"] == 5000
    assert db.generation(owner, active[0])["status"] == "failed"


@pytest.mark.parametrize("kind, suffix, header", [
    ("openai", "/chat/completions", "Authorization"),
    ("anthropic", "/messages", "x-api-key"),
])
def test_compatible_provider_payload(kind, suffix, header, monkeypatch):
    def request(url, *, body, headers):
        assert url == "https://example.com/v1" + suffix
        assert "test-token" in headers[header]
        data = json.loads(body)
        assert data["model"] == "example-model"
        assert data["messages"][-1]["content"] == "input"
        return (b'{"content":[{"type":"text","text":"result"}]}'
                if kind == "anthropic" else b'{"choices":[{"message":{"content":"result"}}]}'), "application/json"

    monkeypatch.setattr(outbound, "request", request)
    assert outbound.Provider(kind, "https://example.com/v1", "example-model", "test-token").complete(
        "system", "input", 12) == "result"


def test_managed_and_user_provider_generations_and_notion_review(setup, monkeypatch):
    db, client, _settings = setup
    owner = account(db)
    credit(db, owner)
    headers = {"Authorization": "Bearer " + db.new_session(owner)}
    used = []

    def complete(provider, _system, prompt, _max_tokens):
        used.append((provider.kind, provider.model))
        source = json.loads(prompt)["chunks"][0]
        return f"- Factual excerpt [{source['citation']}]"

    monkeypatch.setattr(outbound.Provider, "complete", complete)

    def generate(provider, name):
        options = json.dumps({"name": name, "total_gib": 8, "available_gib": 6,
                              "strategy": "compress", "provider": provider,
                              "text": "The orchard grows apples and pears for the autumn harvest."})
        result = client.post("/api/generate", headers=headers, files=[
            ("id", (None, str(uuid4()))), ("options", (None, options)),
            ("files", ("other.md", b"The engineering team works on durable railway tracks.")),
        ]).json()
        assert result["status"] == "completed"
        return result["pack_id"]

    generate({"kind": "managed"}, "Managed build")
    pack = generate({"kind": "anthropic", "model": "user-model", "key": "user-test-secret"}, "User build")
    assert used == [("openai", "managed-test-model"), ("anthropic", "user-model")]
    assert client.get("/api/wallet", headers=headers).json()["balance_mills"] == 5000 - 2 * PRICE_MILLS
    detail = client.get(f"/api/packs/{pack}", headers=headers)
    assert "user-test-secret" not in detail.text
    digest = detail.json()["pack_hash"]
    linked_page = str(uuid4())
    created = []

    def create_page(_self, metadata, pack_digest, sections):
        assert pack_digest == digest and metadata["Name"] == "User build"
        assert any(title == "Knowledge" for title, _text in sections)
        created.append(linked_page)
        return linked_page

    monkeypatch.setattr(NotionStore, "create", create_page)
    body = {"token": "notion-test-secret", "database_id": str(uuid4())}
    notion = f"/api/packs/{pack}/notion"
    for _ in range(2):
        assert client.post(notion, json=body, headers=headers).json()["page_id"] == linked_page
    assert created == [linked_page]
    monkeypatch.setattr(NotionStore, "approval", lambda *_args: ("Approved", "0" * 64))
    assert client.post(notion + "/import", json=body, headers=headers).status_code == 409
    monkeypatch.setattr(NotionStore, "approval", lambda *_args: ("Approved", digest))
    assert client.post(notion + "/import", json=body, headers=headers).json()["status"] == "Approved"
    assert client.get(f"/api/packs/{pack}/export", headers=headers).status_code == 200
