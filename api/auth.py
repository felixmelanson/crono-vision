"""
One-time login: trade CRONO_VISION_TOKEN for an HttpOnly cookie.

The camera page never carries the token in its own JS — it's typed once,
POSTed here, and the browser remembers it via a cookie that JS can't read
back out. That's strictly better than localStorage: nothing to scrape from
devtools, and it isn't subject to iOS's periodic eviction of script-writable
storage.

  GET  /api/auth   -> {"authed": true|false}   (page load: skip the login
                                                  screen if the cookie's
                                                  already valid)
  POST /api/auth   -> {"token": "..."}          sets the cookie on success
  POST /api/auth?logout=1 -> clears it
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webauth import (  # noqa: E402
    clear_cookie_header, expected_token, is_authorized, set_cookie_header,
)

MAX_BODY_BYTES = 4096  # this body is one short string; nothing legitimate is bigger


class handler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        try:
            self._json(200, {"ok": True, "authed": is_authorized(self.headers)})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False, "error": "Internal error."})

    def do_POST(self) -> None:
        try:
            self._handle()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False, "error": "Internal error."})

    def _handle(self) -> None:
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        if params.get("logout"):
            self._json(200, {"ok": True, "authed": False},
                      set_cookie=clear_cookie_header())
            return

        expected = expected_token()
        if not expected:
            self._json(503, {"ok": False, "error": "CRONO_VISION_TOKEN isn't set on the server."})
            return

        length = int(self.headers.get("content-length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"ok": False, "error": "Missing or oversized body."})
            return

        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except ValueError:
            self._json(400, {"ok": False, "error": "Body was not valid JSON."})
            return

        presented = str(payload.get("token") or "").strip()
        if not presented:
            self._json(400, {"ok": False, "error": "Missing 'token'."})
            return

        # Reuse the same check as everywhere else, via a throwaway headers
        # object, so "what counts as the right token" lives in one place.
        if not is_authorized(_FakeHeaders(f"Bearer {presented}")):
            self._json(401, {"ok": False, "error": "Wrong token."})
            return

        self._json(200, {"ok": True, "authed": True}, set_cookie=set_cookie_header(presented))

    def _json(self, status: int, payload: dict, set_cookie: str = None) -> None:
        blob = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        if set_cookie:
            self.send_header("set-cookie", set_cookie)
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class _FakeHeaders:
    """Minimal stand-in so is_authorized() can check a bearer string
    without needing a real HTTPMessage."""

    def __init__(self, auth_value: str):
        self._v = auth_value

    def get(self, name, default=None):
        return self._v if name.lower() == "authorization" else default
