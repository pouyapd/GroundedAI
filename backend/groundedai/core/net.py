"""Network helpers."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})


def is_loopback_url(url: str) -> bool:
    """True if ``url`` points at this machine (localhost, 127.0.0.0/8 or ::1)."""
    host = (urlsplit(url).hostname or "").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def http_client_options(url: str, timeout_s: float) -> dict[str, object]:
    """httpx options for a model server URL.

    Environment proxies (HTTP_PROXY/HTTPS_PROXY) are ignored for loopback
    addresses: a proxy cannot reach a server on this machine's loopback
    interface, and some proxy clients answer such requests with HTTP 503.
    """
    return {"timeout": timeout_s, "trust_env": not is_loopback_url(url)}
