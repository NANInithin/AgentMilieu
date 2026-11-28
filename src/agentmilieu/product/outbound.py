"""Bound the public network calls that use user-supplied destinations."""

import ipaddress
import json
import re
import socket
import ssl
import time
from typing import ClassVar
from urllib.parse import urljoin, urlsplit

import urllib3
from urllib3.exceptions import HTTPError
from urllib3.util import Timeout


class PublicRequestError(ValueError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _public_ip(value):
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            address = address.ipv4_mapped
        elif address.sixtofour or address.teredo or address in ipaddress.ip_network("64:ff9b::/96"):
            return False
    return address.is_global and not address.is_reserved and not address.is_multicast


def _target(url):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port or 443
        if (parsed.scheme != "https" or not host or parsed.username is not None or parsed.fragment
                or not re.fullmatch(r"[a-zA-Z0-9.:-]+", host) or "\\" in url
                or any(ord(char) <= 32 or ord(char) >= 127 for char in url)):
            raise PublicRequestError("Enter a public HTTPS URL without credentials or fragments.")
        results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(item[4][0] for item in results))
        if not addresses or not all(_public_ip(address) for address in addresses):
            raise PublicRequestError("Private and non-public network destinations are unavailable.")
    except (OSError, ValueError) as exc:
        if isinstance(exc, PublicRequestError):
            raise
        raise PublicRequestError("Could not resolve the public HTTPS destination.") from exc
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return parsed, host, port, addresses, path


def request(url, *, body=None, headers=None, limit=1_000_000, redirects=0):
    """Pin the connected address after validation, while TLS and Host use the original name."""
    parsed, host, port, addresses, path = _target(url)
    for address in addresses[:4]:
        pool = urllib3.HTTPSConnectionPool(
            address, port=port, server_hostname=host, assert_hostname=host,
            ssl_context=ssl.create_default_context(), timeout=Timeout(connect=4, read=15),
        )
        try:
            response = pool.urlopen(
                "POST" if body is not None else "GET", path, body=body,
                headers={**(headers or {}), "Host": parsed.netloc, "Accept-Encoding": "identity"},
                retries=False, redirect=False, preload_content=False,
            )
        except (OSError, HTTPError):
            pool.close()
            continue
        try:
            if 300 <= response.status < 400 and redirects and body is None:
                location = response.headers.get("Location")
                if location:
                    return request(urljoin(url, location), limit=limit, redirects=redirects - 1)
            if response.status >= 300:
                raise PublicRequestError(f"The remote service returned HTTP {response.status}.",
                                         status=response.status)
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise PublicRequestError("The remote service returned an unsupported encoding.")
            data = bytearray()
            deadline = time.monotonic() + 25
            while block := response.read1(min(64 << 10, limit + 1 - len(data)), decode_content=False):
                data.extend(block)
                if len(data) > limit:
                    raise PublicRequestError("The remote response exceeds the size limit.")
                if time.monotonic() > deadline:
                    raise PublicRequestError("The remote response exceeded the time limit.")
            return bytes(data), response.headers.get("Content-Type", "").lower()
        except (OSError, HTTPError) as exc:
            raise PublicRequestError("The public HTTPS response could not be read.") from exc
        finally:
            response.close()
            pool.close()
    raise PublicRequestError("Could not connect to the public HTTPS service.")


class Provider:
    DEFAULTS: ClassVar[dict[str, str]] = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1"}

    def __init__(self, kind, base_url, model, key):
        if (kind not in self.DEFAULTS or not model or not key
                or any(not 32 < ord(char) < 127 for char in key)):
            raise PublicRequestError("Choose an API format and supply a model name and key.")
        base = (base_url or self.DEFAULTS[kind]).rstrip("/")
        parts = urlsplit(base)
        if (parts.scheme != "https" or not parts.hostname or parts.username is not None
                or parts.query or parts.fragment):
            raise PublicRequestError("Enter an HTTPS provider base URL without credentials or a query.")
        self.url = base + ("/chat/completions" if kind == "openai" else "/messages")
        self.kind, self.model, self.key = kind, model, key
        self.official_openai = kind == "openai" and parts.hostname == "api.openai.com"

    def complete(self, system, content, max_tokens):
        messages = [{"role": "user", "content": content}]
        limit_field = "max_completion_tokens" if self.official_openai else "max_tokens"
        payload = {"model": self.model, limit_field: max_tokens, "messages": messages}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.kind == "openai":
            headers["Authorization"] = "Bearer " + self.key
            messages.insert(0, {"role": "system", "content": system})
        else:
            headers.update({"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            payload["system"] = system
        try:
            raw, _ = request(self.url, body=json.dumps(payload).encode(), headers=headers)
        except PublicRequestError as exc:
            if self.kind != "openai" or exc.status not in (400, 422):
                raise
            alternate = "max_tokens" if self.official_openai else "max_completion_tokens"
            payload[alternate] = payload.pop(limit_field)
            raw, _ = request(self.url, body=json.dumps(payload).encode(), headers=headers)
        try:
            response = json.loads(raw)
            if self.kind == "anthropic":
                text = "\n".join(block["text"] for block in response["content"] if block["type"] == "text")
            else:
                text = response["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise PublicRequestError("The provider did not return a compatible text response.") from exc
