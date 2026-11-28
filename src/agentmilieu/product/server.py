import asyncio
import hmac
import io
import json
import logging
import secrets
import tempfile
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from psycopg import Error as DatabaseError
from pydantic import BaseModel, Field, SecretStr, ValidationError
from starlette.datastructures import UploadFile
from starlette.middleware.sessions import SessionMiddleware

from ..export import ApprovalError, export
from ..models import MODEL_PRESETS
from ..safety import redact_text
from ..store import NotionStore, canonical, content_hash
from .auth import Google, SignInError
from .config import COOKIE, PRICE_MILLS, TOPUPS, Settings
from .db import Conflict, Database, InsufficientBalance
from .generation import MAX_UPLOAD, GenerationInput, choose_provider, compile_pack, estimate_budget
from .payments import Mollie, PaymentUnavailable

STATIC = Path(__file__).with_name("static")
LOGGER = logging.getLogger(__name__)


class TopupInput(BaseModel):
    id: UUID
    euros: int


class ExchangeInput(BaseModel):
    id: UUID
    secret: SecretStr = Field(min_length=40, max_length=80)


class ReviewInput(BaseModel):
    decision: str = Field(pattern="^(Approved|Rejected)$")
    pack_hash: str = Field(pattern="^[a-f0-9]{64}$")


class NotionInput(BaseModel):
    token: SecretStr = Field(min_length=1, max_length=500)
    database_id: UUID | None = None


def _auth_page(message, body="", status=200):
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in | AgentMilieu</title><link rel="stylesheet" href="/assets/app.css"></head>
<body><main class="auth"><a class="wordmark" href="/">AgentMilieu<span>.</span></a>
<section class="panel"><h1>{escape(message)}</h1>{body}</section></main></body></html>""",
                        status_code=status)


class ExportRegistry:
    def __init__(self, row):
        self.row = row

    def get(self, pack_id):
        return {**self.row, "backend": "local"}

    def load_payload(self, pack_id):
        return self.row["payload"]


class RequestSizeLimit:
    """Apply an aggregate limit before the multipart parser can spool arbitrary data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        limit = MAX_UPLOAD + 512_000 if scope["path"].endswith("/generate") else 16_384
        request = Request(scope)
        length = request.headers.get("content-length", "")
        if length.isdecimal() and int(length) > limit:
            await JSONResponse({"detail": "Request exceeds the size limit."}, 413)(scope, receive, send)
            return
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Request exceeds the size limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None):
    settings = settings or Settings.from_env()
    db = Database(settings.database_url)
    google = Google(settings)
    mollie = Mollie(settings, db)
    generation_slots = BoundedSemaphore(4)

    @asynccontextmanager
    async def lifespan(_app):
        await run_in_threadpool(db.initialize)

        async def maintenance():
            while True:
                try:
                    await run_in_threadpool(db.reap)
                except DatabaseError:
                    LOGGER.error("Database maintenance failed.")
                await asyncio.sleep(60)

        task = asyncio.create_task(maintenance())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestSizeLimit)
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret,
                       session_cookie="milieu_oauth", max_age=600, https_only=settings.secure)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith(("/auth/", "/api/", "/app")):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
                "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": "Check the submitted fields and try again."}, 422)

    def request_token(request):
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer ") and len(bearer) <= 140:
            return bearer[7:], True
        return request.cookies.get(COOKIE, ""), False

    def signed_in(request: Request):
        key, bearer = request_token(request)
        if not key or len(key) > 128 or not (account := db.session_account(key)):
            raise HTTPException(401, "Sign in to continue.")
        if not bearer and request.method != "GET" and request.headers.get("Origin") != settings.public_url:
            raise HTTPException(403, "This request must originate from AgentMilieu.")
        return account

    SignedIn = Annotated[dict, Depends(signed_in)]

    def pack_for(account, pack_id):
        if len(pack_id) != 32 or not (row := db.pack(account["id"], pack_id)):
            raise HTTPException(404, "Context pack not found.")
        return row

    @app.get("/")
    def landing():
        path = STATIC / "index.html"
        if not path.is_file():
            path = Path(__file__).resolve().parents[3] / "website" / "index.html"
        return FileResponse(path)

    @app.get("/healthz")
    def health():
        with db.connect() as conn:
            conn.execute("SELECT 1 FROM accounts LIMIT 1")
        return {"status": "ok"}

    @app.get("/downloads/AgentMilieu-Setup.exe")
    def download_installer():
        if not settings.installer.is_file():
            raise HTTPException(503, "The Windows installer is currently unavailable.")
        return FileResponse(settings.installer, media_type="application/octet-stream",
                            filename="AgentMilieu-Setup.exe",
                            headers={"Cache-Control": "public, max-age=300"})

    @app.head("/downloads/AgentMilieu-Setup.exe")
    def download_installer_head():
        return download_installer()

    @app.get("/app")
    def workspace():
        return FileResponse(STATIC / "workspace.html")

    @app.get("/api/config")
    def config():
        return {"price_mills": PRICE_MILLS, "topups": TOPUPS, "google": google.enabled,
                "mollie": bool(settings.mollie_key), "managed_provider": settings.managed,
                "presets": list(MODEL_PRESETS)}

    @app.get("/auth/google")
    def google_start(request: Request, desktop: UUID | None = None):
        destination = "/app"
        if desktop is not None:
            if db.desktop_login(desktop) is None:
                return _auth_page("This Windows sign-in request expired.", status=400)
            destination = f"/auth/desktop/{desktop}"
        try:
            return RedirectResponse(google.start(request.session, destination), 303)
        except SignInError as exc:
            return _auth_page(str(exc), '<p><a href="/app">Back to AgentMilieu</a></p>', 503)

    @app.get("/auth/google/callback")
    def google_callback(request: Request, state: str = "", code: str = ""):
        try:
            claims, destination = google.finish(request.session, state, code)
        except SignInError as exc:
            return _auth_page(str(exc), '<p><a href="/app">Back to AgentMilieu</a></p>', 400)
        key, _ = request_token(request)
        if key:
            db.end_session(key)
        token = db.new_session(db.account(claims))
        response = RedirectResponse(destination, 303)
        response.set_cookie(COOKIE, token, max_age=7 * 86400, secure=settings.secure,
                            httponly=True, samesite="lax")
        return response

    @app.post("/api/auth/desktop/start")
    def desktop_start():
        if not google.enabled:
            raise HTTPException(503, "Google sign-in is currently unavailable.")
        login_id, secret, code = db.new_desktop_login()
        return {"id": login_id, "secret": secret, "code": code,
                "url": f"{settings.public_url}/auth/desktop/{login_id}"}

    @app.get("/auth/desktop/{login_id}")
    def desktop_confirm(login_id: UUID, request: Request):
        if db.desktop_login(login_id) is None:
            return _auth_page("This Windows sign-in request expired.", status=400)
        try:
            account = signed_in(request)
        except HTTPException:
            return RedirectResponse(f"/auth/google?desktop={login_id}", 303)
        csrf = secrets.token_urlsafe(32)
        request.session["desktop_confirm"] = {"id": str(login_id), "csrf": csrf}
        return _auth_page("Connect the Windows app", f"""<p>Signing in as {escape(account["email"])}.</p>
<p>Only continue if you started sign-in yourself. Enter the six-digit code shown in the Windows app.</p>
<form action="/auth/desktop/{login_id}" method="post">
<input type="hidden" name="csrf" value="{csrf}">
<label>Code<input name="code" pattern="[0-9]{{6}}" inputmode="numeric" autocomplete="off" required></label>
<button class="primary">Connect Windows app</button></form>""")

    @app.post("/auth/desktop/{login_id}")
    async def desktop_approve(login_id: UUID, request: Request, account: SignedIn):
        async with request.form(max_fields=2, max_files=0) as form:
            saved = request.session.pop("desktop_confirm", {})
            csrf, code = form.get("csrf", ""), form.get("code", "")
        if (saved.get("id") != str(login_id) or not isinstance(csrf, str)
                or not hmac.compare_digest(saved.get("csrf", "").encode(), csrf.encode()) or not csrf):
            raise HTTPException(403, "Reopen sign-in from the Windows app.")
        approved = isinstance(code, str) and await run_in_threadpool(
            db.approve_desktop, login_id, account["id"], code)
        if not approved:
            return _auth_page("The code did not match or expired.",
                              f'<p><a href="/auth/desktop/{login_id}">Try again</a></p>', 400)
        return _auth_page("Windows sign-in approved", "<p>Return to the AgentMilieu Windows app.</p>")

    @app.post("/api/auth/desktop/exchange")
    def desktop_exchange(form: ExchangeInput):
        token = db.exchange_desktop(form.id, form.secret.get_secret_value())
        return {"token": token} if token else JSONResponse({"status": "pending"}, 202)

    @app.get("/api/me")
    def me(account: SignedIn):
        return account

    @app.post("/api/logout")
    def logout(request: Request, account: SignedIn):
        db.end_session(request_token(request)[0])
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, secure=settings.secure, httponly=True, samesite="lax")
        return response

    @app.get("/api/wallet")
    def wallet(account: SignedIn):
        return db.wallet(account["id"])

    @app.post("/api/wallet/topups")
    def checkout(form: TopupInput, account: SignedIn):
        if form.euros not in TOPUPS:
            raise HTTPException(422, "Choose €5, €10, or €20.")
        try:
            return {"url": mollie.checkout(account["id"], form.id, form.euros), "id": form.id}
        except (Conflict, PaymentUnavailable) as exc:
            raise HTTPException(409 if isinstance(exc, Conflict) else 503, str(exc)) from exc

    @app.post("/api/wallet/topups/{topup_id}/refresh")
    def refresh_topup(topup_id: UUID, account: SignedIn):
        topup = db.topup(account["id"], topup_id)
        if not topup:
            raise HTTPException(404, "Top-up not found.")
        if topup["mollie_id"]:
            try:
                mollie.refresh(topup["mollie_id"], topup_id)
            except PaymentUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
        return db.wallet(account["id"])

    @app.post("/api/mollie/webhook/{topup_id}")
    async def webhook(topup_id: UUID, request: Request):
        async with request.form(max_files=0, max_fields=3) as form:
            payment_id = form.get("id", "")
        try:
            if isinstance(payment_id, str):
                await run_in_threadpool(mollie.refresh, payment_id, topup_id)
        except PaymentUnavailable as exc:
            raise HTTPException(503, "Payment verification temporarily unavailable.") from exc
        return Response(status_code=200)

    @app.get("/api/generations/{generation_id}")
    def generation_status(generation_id: UUID, account: SignedIn):
        row = db.generation(account["id"], generation_id)
        if row is None:
            raise HTTPException(404, "Generation not found.")
        return row

    @app.post("/api/budget")
    def budget(form: GenerationInput, account: SignedIn):
        try:
            return estimate_budget(form)
        except ValueError as exc:
            raise HTTPException(422, redact_text(str(exc))[0]) from exc

    def generate_pack(account_id, generation_id, form, uploads):
        try:
            provider = choose_provider(form.provider, settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not generation_slots.acquire(blocking=False):
            raise HTTPException(503, "All generation slots are busy. Try again shortly.")
        try:
            try:
                existing = db.reserve_generation(account_id, generation_id)
            except InsufficientBalance as exc:
                raise HTTPException(402, str(exc)) from exc
            except Conflict as exc:
                raise HTTPException(409, str(exc)) from exc
            if existing:
                return existing
            try:
                pack = compile_pack(form, uploads, provider)
            except ValueError as exc:
                db.finish_generation(account_id, generation_id, error=redact_text(str(exc))[0][:400])
            except Exception:  # noqa: BLE001 -- release the charge on every compiler failure
                LOGGER.error("Unexpected generation failure for %s.", generation_id)
                db.finish_generation(account_id, generation_id,
                                     error="Context generation failed. Your wallet was credited.")
            else:
                db.finish_generation(account_id, generation_id, pack=pack)
            return db.generation(account_id, generation_id)
        finally:
            generation_slots.release()

    @app.post("/api/generate")
    async def generate(request: Request, account: SignedIn):
        async with request.form(max_files=8, max_fields=2, max_part_size=510_000) as multipart:
            try:
                generation_id = UUID(multipart.get("id", ""))
                form = GenerationInput.model_validate_json(multipart.get("options", ""))
            except (ValidationError, ValueError, TypeError, AttributeError) as exc:
                raise HTTPException(422, "Check the source, model, provider, and memory fields.") from exc
            uploads, total = [], 0
            for item in multipart.getlist("files"):
                if not isinstance(item, UploadFile) or not item.filename:
                    raise HTTPException(422, "Select valid source files.")
                data = await item.read(MAX_UPLOAD - total + 1)
                total += len(data)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, "The combined source files exceed 12 MiB.")
                uploads.append((item.filename, data))
        if not (uploads or form.text.strip() or form.urls):
            raise HTTPException(422, "Add a document, text, or a public HTTPS source.")
        row = await run_in_threadpool(generate_pack, account["id"], generation_id, form, uploads)
        return row

    @app.get("/api/packs")
    def packs(account: SignedIn, offset: int = 0):
        if not 0 <= offset <= 100_000:
            raise HTTPException(422, "Invalid history page.")
        rows = db.packs(account["id"], offset)
        return {"items": rows[:50], "has_more": len(rows) > 50}

    @app.get("/api/packs/{pack_id}")
    def pack_detail(pack_id: str, account: SignedIn):
        row = pack_for(account, pack_id)
        payload = row.pop("payload")
        return {**row, "metadata": payload["metadata"], "pack": payload["pack"],
                "audit": payload["files"]["audit.md"], "safety": json.loads(payload["files"]["safety.json"])}

    @app.post("/api/packs/{pack_id}/review")
    def review(pack_id: str, form: ReviewInput, account: SignedIn):
        row = pack_for(account, pack_id)
        if not (hmac.compare_digest(form.pack_hash, row["pack_hash"])
                and hmac.compare_digest(content_hash(row["payload"]), row["pack_hash"])):
            raise HTTPException(409, "The reviewed content changed. Refresh the pack.")
        return db.review(account["id"], pack_id, form.decision, form.pack_hash)

    @app.get("/api/packs/{pack_id}/export")
    def download_pack(pack_id: str, account: SignedIn):
        row = pack_for(account, pack_id)
        with tempfile.TemporaryDirectory(prefix="milieu-export-") as destination:
            try:
                path = export(pack_id, registry=ExportRegistry(row), destination=destination)
            except ApprovalError as exc:
                raise HTTPException(409, str(exc)) from exc
            output = io.BytesIO()
            with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
                for artifact in sorted(path.iterdir()):
                    archive.write(artifact, artifact.name)
        return Response(output.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="AgentMilieu-{pack_id}.zip"'})

    def notion_store(form):
        from notion_client import Client

        client = Client(auth=form.token.get_secret_value(), notion_version="2022-06-28", timeout_ms=8000)
        return NotionStore(str(form.database_id) if form.database_id else "unused",
                           client=client, retries=0)

    @app.post("/api/packs/{pack_id}/notion")
    def publish_notion(pack_id: str, form: NotionInput, account: SignedIn):
        def create_page(row):
            if not form.database_id:
                raise HTTPException(422, "Enter the Notion database ID.")
            payload = row["payload"]
            if not hmac.compare_digest(content_hash(payload), row["pack_hash"]):
                raise HTTPException(409, "The pack hash changed.")
            pack = payload["pack"]
            try:
                return notion_store(form).create(payload["metadata"], row["pack_hash"], [
                    ("System prompt", pack["system_prompt"]), ("Knowledge", pack["knowledge"]),
                    ("Audit", payload["files"]["audit.md"]), ("Sources", canonical(pack["sources"])),
                ])
            except Exception as exc:
                raise HTTPException(502, "Could not publish. Check the Notion token, database access, and schema.") from exc

        page_id = db.link_notion(account["id"], pack_id, create_page)
        if not page_id:
            raise HTTPException(404, "Context pack not found.")
        return {"page_id": page_id}

    @app.post("/api/packs/{pack_id}/notion/import")
    def import_notion(pack_id: str, form: NotionInput, account: SignedIn):
        row = pack_for(account, pack_id)
        if not row["notion_page_id"]:
            raise HTTPException(409, "Send this pack to Notion first.")
        try:
            decision, digest = notion_store(form).approval(row["notion_page_id"])
        except Exception as exc:
            raise HTTPException(502, "Could not verify the Notion decision.") from exc
        if decision not in ("Approved", "Rejected"):
            raise HTTPException(409, "Approve or reject the linked pack in Notion first.")
        if not (hmac.compare_digest(digest.encode(), row["pack_hash"].encode())
                and hmac.compare_digest(content_hash(row["payload"]), row["pack_hash"])):
            raise HTTPException(409, "The Notion decision does not match this pack hash.")
        return db.review(account["id"], pack_id, decision, digest)

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
    return app
