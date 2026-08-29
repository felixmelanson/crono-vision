"""
Shared auth for the two Vercel endpoints (api/index.py, api/auth.py).

Two ways in, same secret:
  - Authorization: Bearer <token>   — what the iOS Shortcut sends.
  - Cookie: cv_auth=<token>         — what the browser page sends, set once
                                       by /api/auth so the token never lives
                                       in JS or gets bundled into the page.

The cookie holds the token itself rather than a signed session — there's one
user and one secret, so a session layer would be complexity with nothing to
protect against that CRONO_VISION_TOKEN doesn't already cover. Rotate the
token (change the env var) and every cookie stops working, which is exactly
"log everyone out."
"""

from __future__ import annotations

import hmac
import os
from http.cookies import SimpleCookie
from typing import Optional

COOKIE_NAME = "cv_auth"
COOKIE_MAX_AGE = 365 * 24 * 3600  # a year — this is a home-screen app, not a website


def expected_token() -> Optional[str]:
    return os.environ.get("CRONO_VISION_TOKEN") or None


def is_authorized(headers) -> bool:
    """headers: anything with .get(name) -> str|None, case-insensitive
    (http.client.HTTPMessage from BaseHTTPRequestHandler qualifies)."""
    expected = expected_token()
    if not expected:
        return False  # fail closed if the env var never got set

    auth = headers.get("authorization") or ""
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    if bearer and hmac.compare_digest(bearer.strip(), expected):
        return True

    cookie_header = headers.get("cookie") or ""
    if cookie_header:
        jar = SimpleCookie()
        try:
            jar.load(cookie_header)
        except Exception:
            return False
        morsel = jar.get(COOKIE_NAME)
        if morsel and hmac.compare_digest(morsel.value.strip(), expected):
            return True

    return False


def set_cookie_header(value: str, max_age: int = COOKIE_MAX_AGE) -> str:
    """Value for a Set-Cookie response header that logs the browser in."""
    return (
        f"{COOKIE_NAME}={value}; Max-Age={max_age}; Path=/; "
        f"HttpOnly; Secure; SameSite=Strict"
    )


def clear_cookie_header() -> str:
    """Value for a Set-Cookie response header that logs the browser out."""
    return f"{COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; Secure; SameSite=Strict"
