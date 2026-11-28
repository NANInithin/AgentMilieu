import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import requests

from .db import Conflict


class PaymentUnavailable(ValueError):
    pass


PAYMENT_ID = re.compile(r"tr_[A-Za-z0-9]{1,100}")
STATUSES = {"open", "pending", "authorized", "paid", "canceled", "expired", "failed"}


def _mollie_amount(value):
    if value.get("currency") != "EUR" or not re.fullmatch(r"\d{1,8}\.\d{2}", value.get("value", "")):
        raise ValueError("Invalid payment amount")
    return int(Decimal(value["value"]) * 1000)


class Mollie:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db

    def _request(self, method, path, *, payload=None, key=None):
        if not self.settings.mollie_key:
            raise PaymentUnavailable("Wallet top-ups are currently unavailable.")
        headers = {"Authorization": "Bearer " + self.settings.mollie_key}
        if key:
            headers["Idempotency-Key"] = str(key)
        try:
            response = requests.request(
                method, "https://api.mollie.com/v2/payments" + path, json=payload,
                headers=headers, timeout=(5, 20), allow_redirects=False,
            )
            if not response.ok or response.is_redirect:
                raise PaymentUnavailable("Mollie could not complete the request. Please retry.")
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if isinstance(exc, PaymentUnavailable):
                raise
            raise PaymentUnavailable("Mollie could not complete the request. Please retry.") from exc

    def checkout(self, account_id, topup_id, euros):
        if not self.settings.mollie_key:
            raise PaymentUnavailable("Wallet top-ups are currently unavailable.")
        topup = self.db.start_topup(account_id, topup_id, euros * 1000)
        if topup["mollie_id"]:
            if topup["status"] not in ("open", "pending", "authorized") or not topup["checkout_url"]:
                raise Conflict("This payment is closed. Start a new top-up.")
            return topup["checkout_url"]
        origin = self.settings.public_url
        snapshot_at = datetime.now(timezone.utc)
        if snapshot_at - topup["created_at"] >= timedelta(minutes=55):
            raise Conflict("This checkout attempt expired. Start a new top-up.")
        result = self._request("POST", "", key=topup_id, payload={
            "amount": {"currency": "EUR", "value": f"{euros}.00"},
            "description": f"AgentMilieu wallet top-up {topup_id}",
            "redirectUrl": f"{origin}/app?topup={topup_id}",
            **({"webhookUrl": f"{origin}/api/mollie/webhook/{topup_id}"} if self.settings.secure else {}),
            "metadata": {"agentmilieu_topup": str(topup_id)},
        })
        if not self.apply(result, expected=topup_id, require_checkout=True, snapshot_at=snapshot_at):
            raise PaymentUnavailable("Mollie returned a payment that could not be matched.")
        return result["_links"]["checkout"]["href"]

    def refresh(self, payment_id, topup_id):
        if PAYMENT_ID.fullmatch(payment_id) and self.db.matches_topup(topup_id, payment_id):
            snapshot_at = datetime.now(timezone.utc)
            return self.apply(self._request("GET", "/" + payment_id), payment_id=payment_id,
                              expected=topup_id, snapshot_at=snapshot_at)
        return False

    def apply(self, payment, *, expected=None, payment_id=None, require_checkout=False, snapshot_at=None):
        try:
            topup_id = UUID(payment["metadata"]["agentmilieu_topup"])
            pid, status = payment["id"], payment["status"]
            if (
                (expected and expected != topup_id) or (payment_id and pid != payment_id)
                or not PAYMENT_ID.fullmatch(pid) or status not in STATUSES
            ):
                return False
            amount = _mollie_amount(payment["amount"])
            reversed_amount = sum(
                _mollie_amount(payment[field]) for field in ("amountRefunded", "amountChargedBack")
                if payment.get(field) is not None
            )
            checkout = (payment.get("_links", {}).get("checkout") or {}).get("href")
            if checkout and not checkout.startswith("https://"):
                return False
            if require_checkout and not checkout:
                return False
        except (AttributeError, KeyError, ValueError, TypeError):
            return False
        return self.db.apply_payment(topup_id, pid, amount, status, checkout,
                                     min(amount, reversed_amount), snapshot_at)
