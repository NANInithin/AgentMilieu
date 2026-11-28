import base64
import hashlib
import secrets
from urllib.parse import urlencode

import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token


class SignInError(ValueError):
    pass


class VerificationRequest(Request):
    def __call__(self, *args, **kwargs):
        kwargs["timeout"] = 8
        return super().__call__(*args, **kwargs)


class Google:
    def __init__(self, settings):
        self.settings = settings
        self.redirect_uri = settings.public_url + "/auth/google/callback"

    @property
    def enabled(self):
        return bool(self.settings.google_id and self.settings.google_secret)

    def start(self, session, destination):
        if not self.enabled:
            raise SignInError("Google sign-in is currently unavailable.")
        flow = {field: secrets.token_urlsafe(32) for field in ("state", "nonce", "verifier")}
        session["google"] = {**flow, "destination": destination}
        challenge = base64.urlsafe_b64encode(hashlib.sha256(flow["verifier"].encode()).digest())
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "client_id": self.settings.google_id, "redirect_uri": self.redirect_uri,
            "scope": "openid email profile", "response_type": "code", "state": flow["state"],
            "nonce": flow["nonce"], "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        })

    def finish(self, session, state, code):
        flow = session.pop("google", None)
        if not flow or not code or not secrets.compare_digest(flow["state"].encode(), state.encode()):
            raise SignInError("This sign-in request expired. Start again.")
        try:
            response = requests.post("https://oauth2.googleapis.com/token", data={
                "code": code, "client_id": self.settings.google_id,
                "client_secret": self.settings.google_secret, "redirect_uri": self.redirect_uri,
                "code_verifier": flow["verifier"], "grant_type": "authorization_code",
            }, timeout=(5, 10), allow_redirects=False)
            response.raise_for_status()
            claims = id_token.verify_oauth2_token(response.json()["id_token"], VerificationRequest(),
                                                  self.settings.google_id)
            if not (claims.get("nonce") == flow["nonce"] and claims.get("email_verified") is True
                    and claims.get("sub") and claims.get("email")):
                raise ValueError("Missing or unverified identity")
            if not self.enabled or not isinstance(claims["email"], str):
                raise ValueError("Invalid account")
        except Exception as exc:
            raise SignInError("Google could not verify the sign-in. Try again.") from exc
        return claims, flow["destination"]
