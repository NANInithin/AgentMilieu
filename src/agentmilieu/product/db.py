import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..store import content_hash
from .config import PRICE_MILLS


class Conflict(ValueError):
    pass


class InsufficientBalance(Conflict):
    pass


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Database:
    def __init__(self, url):
        self.url = url

    def connect(self):
        return psycopg.connect(self.url, row_factory=dict_row, connect_timeout=5)

    def initialize(self):
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(247031859)")
            conn.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))

    def account(self, claims):
        with self.connect() as conn:
            return conn.execute("""
                INSERT INTO accounts (id, google_sub, email, name) VALUES (%s, %s, %s, %s)
                ON CONFLICT (google_sub) DO UPDATE SET email = excluded.email, name = excluded.name
                RETURNING id
            """, (uuid4(), claims["sub"], claims["email"], claims.get("name") or claims["email"])
            ).fetchone()["id"]

    def _session(self, conn, account_id, days):
        token = secrets.token_urlsafe(32)
        conn.execute("""
            INSERT INTO sessions VALUES (%s, %s, now() + make_interval(days => %s))
        """, (token_hash(token), account_id, days))
        return token

    def new_session(self, account_id):
        with self.connect() as conn:
            return self._session(conn, account_id, 7)

    def session_account(self, token):
        with self.connect() as conn:
            return conn.execute("""
                SELECT a.id, a.email, a.name, a.balance_mills FROM accounts a
                JOIN sessions s ON s.account_id = a.id
                WHERE s.token_hash = %s AND s.expires_at > now()
            """, (token_hash(token),)).fetchone()

    def end_session(self, token):
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = %s", (token_hash(token),))

    def new_desktop_login(self):
        login_id, secret = uuid4(), secrets.token_urlsafe(32)
        code = f"{secrets.randbelow(10 ** 6):06d}"
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO desktop_logins (id, secret_hash, code, expires_at)
                VALUES (%s, %s, %s, now() + interval '10 minutes')
            """, (login_id, token_hash(secret), code))
        return login_id, secret, code

    def desktop_login(self, login_id):
        with self.connect() as conn:
            return conn.execute("""
                SELECT code FROM desktop_logins
                WHERE id = %s AND account_id IS NULL AND expires_at > now()
            """, (login_id,)).fetchone()

    def approve_desktop(self, login_id, account_id, code):
        with self.connect() as conn:
            return conn.execute("""
                UPDATE desktop_logins SET account_id = %s
                WHERE id = %s AND code = %s AND account_id IS NULL AND expires_at > now()
            """, (account_id, login_id, code)).rowcount == 1

    def exchange_desktop(self, login_id, secret):
        with self.connect() as conn:
            row = conn.execute("""
                DELETE FROM desktop_logins WHERE id = %s AND secret_hash = %s
                    AND account_id IS NOT NULL AND expires_at > now()
                RETURNING account_id
            """, (login_id, token_hash(secret))).fetchone()
            return self._session(conn, row["account_id"], 30) if row else None

    @staticmethod
    def _wallet_entry(conn, account_id, kind, reference, amount):
        conn.execute("""
            INSERT INTO wallet_entries (account_id, kind, reference, amount_mills)
            VALUES (%s, %s, %s, %s)
        """, (account_id, kind, str(reference), amount))
        conn.execute("UPDATE accounts SET balance_mills = balance_mills + %s WHERE id = %s",
                     (amount, account_id))

    def wallet(self, account_id):
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            balance = conn.execute("SELECT balance_mills FROM accounts WHERE id = %s",
                                   (account_id,)).fetchone()["balance_mills"]
            entries = conn.execute("""
                SELECT kind, reference, amount_mills, created_at FROM wallet_entries
                WHERE account_id = %s ORDER BY created_at DESC LIMIT 100
            """, (account_id,)).fetchall()
            topups = conn.execute("""
                SELECT id, amount_mills, status, created_at FROM topups
                WHERE account_id = %s ORDER BY created_at DESC LIMIT 20
            """, (account_id,)).fetchall()
            return {"balance_mills": balance, "entries": entries, "topups": topups}

    def start_topup(self, account_id, topup_id, amount):
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO topups (id, account_id, amount_mills) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (topup_id, account_id, amount))
            topup = conn.execute("SELECT * FROM topups WHERE id = %s AND account_id = %s",
                                 (topup_id, account_id)).fetchone()
            if topup is None or topup["amount_mills"] != amount:
                raise Conflict("This top-up reference was already used.")
            return topup

    def topup(self, account_id, topup_id):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM topups WHERE id = %s AND account_id = %s",
                                (topup_id, account_id)).fetchone()

    def matches_topup(self, topup_id, payment_id):
        with self.connect() as conn:
            return bool(conn.execute("""
                SELECT 1 FROM topups WHERE id = %s AND (mollie_id IS NULL OR mollie_id = %s)
            """, (topup_id, payment_id)).fetchone())

    def apply_payment(self, topup_id, payment_id, amount, status, checkout, reversal, snapshot_at=None):
        snapshot_at = snapshot_at or datetime.now(timezone.utc)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM topups WHERE id = %s FOR UPDATE", (topup_id,)).fetchone()
            if not row or row["mollie_id"] not in (None, payment_id) or row["amount_mills"] != amount:
                return False
            if row["mollie_snapshot_at"] and snapshot_at <= row["mollie_snapshot_at"]:
                return True
            account_id = row["account_id"]
            credited, reversed_amount = row["credited_mills"], row["reversed_mills"]
            if status == "paid" and credited == 0:
                self._wallet_entry(conn, account_id, "Top-up", topup_id, amount)
                credited = amount
            if credited and reversal != reversed_amount:
                adjustment = reversed_amount - reversal
                kind = "Payment reversal" if adjustment < 0 else "Reversal adjustment"
                self._wallet_entry(conn, account_id, kind, uuid4(), adjustment)
                reversed_amount = reversal
            if reversed_amount:
                status = "reversed" if reversed_amount == amount else "partially reversed"
            elif credited:
                status = "paid"
            elif row["status"] in ("canceled", "expired", "failed") and status in (
                "open", "pending", "authorized"
            ):
                status = row["status"]
            conn.execute("""
                UPDATE topups SET mollie_id = %s, status = %s, checkout_url = coalesce(%s, checkout_url),
                    credited_mills = %s, reversed_mills = %s, mollie_snapshot_at = %s WHERE id = %s
            """, (payment_id, status, checkout, credited, reversed_amount, snapshot_at, topup_id))
            return True

    def generation(self, account_id, generation_id):
        with self.connect() as conn:
            return conn.execute("""
                SELECT id, status, pack_id, error FROM generations WHERE account_id = %s AND id = %s
            """, (account_id, generation_id)).fetchone()

    def _release_expired(self, conn, account_id):
        stale = conn.execute("""
            UPDATE generations SET status = 'failed', error = 'Generation timed out. Credits returned.'
            WHERE account_id = %s AND status = 'running' AND lease_until <= now() RETURNING id
        """, (account_id,)).fetchone()
        if stale:
            self._wallet_entry(conn, account_id, "Generation credit", stale["id"], PRICE_MILLS)

    def reserve_generation(self, account_id, generation_id):
        with self.connect() as conn:
            conn.execute("SELECT id FROM accounts WHERE id = %s FOR UPDATE", (account_id,))
            self._release_expired(conn, account_id)
            row = conn.execute("""
                SELECT id, status, pack_id, error FROM generations WHERE account_id = %s AND id = %s
            """, (account_id, generation_id)).fetchone()
            if row:
                return row
            if conn.execute("""
                SELECT 1 FROM generations WHERE account_id = %s AND status = 'running'
            """, (account_id,)).fetchone():
                raise Conflict("A generation is already in progress.")
            bal = conn.execute("SELECT balance_mills FROM accounts WHERE id = %s",
                               (account_id,)).fetchone()["balance_mills"]
            if bal < PRICE_MILLS:
                raise InsufficientBalance("Top up your wallet to generate a context pack.")
            conn.execute("INSERT INTO generations (account_id, id, status) VALUES (%s, %s, 'running')",
                         (account_id, generation_id))
            self._wallet_entry(conn, account_id, "Context generation", generation_id, -PRICE_MILLS)
            return None

    def finish_generation(self, account_id, generation_id, pack=None, error=None):
        with self.connect() as conn:
            conn.execute("SELECT id FROM accounts WHERE id = %s FOR UPDATE", (account_id,))
            self._release_expired(conn, account_id)
            row = conn.execute("""
                SELECT status FROM generations WHERE account_id = %s AND id = %s
            """, (account_id, generation_id)).fetchone()
            if row["status"] != "running":
                return
            pack_id = None
            if pack:
                pack_id, payload = pack
                conn.execute("""
                    INSERT INTO packs (id, account_id, payload, pack_hash) VALUES (%s, %s, %s, %s)
                """, (pack_id, account_id, Jsonb(payload), content_hash(payload)))
            else:
                self._wallet_entry(conn, account_id, "Generation credit", generation_id, PRICE_MILLS)
            conn.execute("""
                UPDATE generations SET status = %s, pack_id = %s, error = %s
                WHERE account_id = %s AND id = %s
            """, ("completed" if pack else "failed", pack_id, error, account_id, generation_id))

    def reap(self):
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= now()")
            conn.execute("DELETE FROM desktop_logins WHERE expires_at <= now()")
            overdue = conn.execute("""
                SELECT account_id FROM generations WHERE status = 'running' AND lease_until <= now()
            """).fetchall()
        for row in overdue:
            with self.connect() as conn:
                account_id = row["account_id"]
                conn.execute("SELECT id FROM accounts WHERE id = %s FOR UPDATE", (account_id,))
                self._release_expired(conn, account_id)

    def packs(self, account_id, offset):
        with self.connect() as conn:
            return conn.execute("""
                SELECT id, pack_hash, status, notion_page_id, created_at,
                    payload->'metadata' AS metadata
                FROM packs WHERE account_id = %s ORDER BY created_at DESC, id DESC
                LIMIT 51 OFFSET %s
            """, (account_id, offset)).fetchall()

    def pack(self, account_id, pack_id):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM packs WHERE id = %s AND account_id = %s",
                                (pack_id, account_id)).fetchone()

    def review(self, account_id, pack_id, status, digest):
        with self.connect() as conn:
            return conn.execute("""
                UPDATE packs SET status = %s, approved_hash = %s
                WHERE id = %s AND account_id = %s AND pack_hash = %s RETURNING status
            """, (status, digest if status == "Approved" else None, pack_id, account_id, digest)
            ).fetchone()

    def link_notion(self, account_id, pack_id, create_page):
        with self.connect() as conn:
            row = conn.execute("""
                SELECT * FROM packs WHERE id = %s AND account_id = %s FOR UPDATE
            """, (pack_id, account_id)).fetchone()
            if not row or row["notion_page_id"]:
                return row["notion_page_id"] if row else None
            page_id = create_page(row)
            conn.execute("UPDATE packs SET notion_page_id = %s WHERE id = %s", (page_id, pack_id))
            return page_id
