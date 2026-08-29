"""
Vercel serverless endpoint: POST a food photo, get diary entries.

Uses BaseHTTPRequestHandler rather than FastAPI so the deployed bundle stays
at three dependencies. Vercel's Python builder only auto-detects a handful
of filenames in api/ (app.py, index.py, server.py, main.py, wsgi.py,
asgi.py) — index.py is the one that keeps this file unconfigured by any
pyproject.toml, and as the "index" file it maps to /api rather than
/api/index.

The iOS Shortcut can send the photo either way:

  1. Raw bytes  — "Get Contents of URL", method POST, request body = File.
                  Put the meal/date/hint in the query string.
  2. JSON       — {"image": "<base64>", "meal": "lunch", "hint": "...",
                   "date": "2026-08-28", "dry_run": false}

The browser camera page authenticates a different way — see webauth.py — by
sending the cookie /api/auth sets after a one-time login, so the token never
lives in that page's own JS. Both this file and that one check
webauth.is_authorized(), which accepts either.

A plain POST runs the whole thing and answers once, which is what the
Shortcut wants. The camera page splits it in two so it can show the person
their food was recognised without waiting on the diary writes:

  POST   /api?phase=analyze              — vision + matching, writes nothing.
                                            Returns `pending`, each item
                                            carrying the plan to write.
  POST   /api?action=commit  (JSON body) — writes those plans. Body is
                                            {date, meal, items: [...]}.

This file also handles the two actions the capture page needs after a log:
  DELETE /api?entry_id=X&date=Y          — undo: remove one diary entry.
  POST   /api?action=swap  (JSON body)   — swap: remove one entry, add another
                                            in its place (the "not this? try
                                            the runner-up" flow).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Vercel puts the function's own directory on sys.path, not the repo root,
# so the shared modules one level up need an explicit nudge.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import analyze, commit, log_photo  # noqa: E402
from vision import VisionError  # noqa: E402
from cronometer_client import CronometerAuthError, CronometerClient, CronometerError  # noqa: E402
from webauth import is_authorized  # noqa: E402

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # comfortably above an iPhone photo
# A commit body is analyze's own output handed back, match alternatives
# and all — small, but not swap-sized.
MAX_COMMIT_BYTES = 256 * 1024


class handler(BaseHTTPRequestHandler):
    # Vercel's Python runtime looks for a class named exactly `handler`.

    def do_POST(self) -> None:
        try:
            params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if params.get("action") == "swap":
                self._handle_swap()
            elif params.get("action") == "commit":
                self._handle_commit()
            else:
                self._handle_log(analyze_only=params.get("phase") == "analyze")
        except Exception:  # noqa: BLE001 - last line of defence
            traceback.print_exc()
            self._json(500, {"ok": False, "error": "Internal error. Check the function logs."})

    def do_DELETE(self) -> None:
        try:
            self._handle_undo()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False, "error": "Internal error. Check the function logs."})

    def do_GET(self) -> None:
        """Unauthenticated: health check, so you can confirm a deploy from a
        browser. Authenticated + ?daily=1: today's running totals, for the
        camera page's daily-total pill — kept behind auth since it's real
        diary data, unlike the health check's config booleans."""
        try:
            params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if params.get("daily") and is_authorized(self.headers):
                self._handle_daily(params.get("date"))
                return
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False, "error": "Internal error. Check the function logs."})
            return

        self._json(200, {
            "ok": True,
            "service": "crono-vision",
            "usage": "POST an image here with Authorization: Bearer <token>",
            "configured": {
                "cronometer": bool(os.environ.get("CRONOMETER_EMAIL")
                                   and os.environ.get("CRONOMETER_PASSWORD")),
                "gemini": bool(os.environ.get("GEMINI_API_KEY")),
                "auth_token": bool(os.environ.get("CRONO_VISION_TOKEN")),
                "timezone": os.environ.get("CRONOMETER_TIMEZONE") or "(unset — will use UTC)",
            },
        })

    def _handle_daily(self, date: str = None) -> None:
        client = CronometerClient()
        try:
            daily = client.get_daily_nutrition(date)
        except CronometerAuthError as e:
            self._json(401, {"ok": False, "error": str(e), "stage": "cronometer_auth"})
            return
        except CronometerError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "cronometer"})
            return
        finally:
            client.close()
        daily.pop("nutrient_targets", None)
        daily.pop("entries", None)
        self._json(200, {"ok": True, "daily": daily})

    # -- request handling ---------------------------------------------

    def _handle_log(self, analyze_only: bool = False) -> None:
        if not is_authorized(self.headers):
            self._json(401, {"ok": False, "error": "Unauthorized"})
            return

        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            self._json(400, {"ok": False, "error": "Empty request body"})
            return
        if length > MAX_IMAGE_BYTES:
            self._json(413, {"ok": False, "error": f"Image over {MAX_IMAGE_BYTES // 1024 // 1024}MB"})
            return

        body = self.rfile.read(length)
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        content_type = (self.headers.get("content-type") or "").split(";")[0].strip().lower()

        if content_type == "application/json":
            try:
                payload = json.loads(body)
            except ValueError:
                self._json(400, {"ok": False, "error": "Body was not valid JSON"})
                return
            image = payload.get("image") or payload.get("photo")
            if not image:
                self._json(400, {"ok": False, "error": "JSON body needs an 'image' field (base64)"})
                return
            try:
                image = base64.b64decode(str(image).split(",")[-1], validate=False)
            except Exception:
                self._json(400, {"ok": False, "error": "'image' was not valid base64"})
                return
            params = {**params, **{k: v for k, v in payload.items() if k not in ("image", "photo")}}
        else:
            # Anything else is treated as the raw image, which is what
            # Shortcuts sends when the body is a file.
            image = body

        common = {
            "date": params.get("date"),
            "meal": params.get("meal"),
            "hint": params.get("hint") or params.get("note"),
            # Off by default (the Shortcut's original conservative
            # behavior); the camera page passes always_log=true since
            # its undo/swap UI is what makes logging-while-unsure safe.
            "always_log_uncertain": _truthy(params.get("always_log")),
        }

        try:
            if analyze_only:
                # Phase one: everything up to the writes. The camera page
                # takes this so it can draw its detection markers while the
                # diary writes are still in flight, instead of holding a
                # frozen frame until the whole thing is done.
                result = analyze(image, **common)
            else:
                result = log_photo(
                    image,
                    dry_run=_truthy(params.get("dry_run")),
                    **common,
                )
        except VisionError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "vision"})
            return
        except CronometerAuthError as e:
            self._json(401, {"ok": False, "error": str(e), "stage": "cronometer_auth"})
            return
        except CronometerError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "cronometer"})
            return
        except ValueError as e:
            self._json(400, {"ok": False, "error": str(e)})
            return

        # `summary` first and flat, so a Shortcut can read it without
        # digging through nested JSON to show a notification. The analyze
        # phase has no summary to give — nothing has happened yet.
        head = {"ok": True}
        if "summary" in result:
            head["summary"] = result["summary"]
        self._json(200, {**head, **result})

    def _handle_commit(self) -> None:
        """POST /api?action=commit — phase two: write what analyze resolved.

        Body: {date, meal, items: [<pending entry from analyze>, ...]}. The
        entries go back out whole, so the page keeps the guess and match
        data it already has and only gains the entry ids.

        The plans are trusted the same way swap's are: this is the account
        owner's own session writing to their own diary, and a food_id is
        not a capability. What is checked is that each one is well-formed,
        so a mangled body fails here rather than halfway through a batch.
        """
        if not is_authorized(self.headers):
            self._json(401, {"ok": False, "error": "Unauthorized"})
            return

        length = int(self.headers.get("content-length") or 0)
        if length <= 0 or length > MAX_COMMIT_BYTES:
            self._json(400, {"ok": False, "error": "Missing or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._json(400, {"ok": False, "error": "Body was not valid JSON"})
            return

        items = body.get("items")
        if not isinstance(items, list):
            self._json(400, {"ok": False, "error": "Body needs an 'items' list"})
            return
        if not items:
            self._json(200, {"ok": True, "logged": [], "failed": []})
            return

        plans = []
        for item in items:
            plan = (item or {}).get("plan") or {}
            try:
                clean = {
                    "food_id": int(plan["food_id"]),
                    "measure_id": int(plan["measure_id"]),
                    "grams": float(plan["grams"]),
                    "food_name": str(plan.get("food_name") or ""),
                }
            except (AttributeError, KeyError, TypeError, ValueError):
                self._json(400, {"ok": False,
                                 "error": "Each item needs a plan with food_id, "
                                          "measure_id and grams"})
                return
            if clean["grams"] <= 0:
                self._json(400, {"ok": False, "error": "grams must be positive"})
                return
            plans.append({**item, "plan": clean})

        client = CronometerClient()
        try:
            logged, failed = commit(plans, date=body.get("date"),
                                    meal=body.get("meal"), client=client)
        except CronometerAuthError as e:
            self._json(401, {"ok": False, "error": str(e), "stage": "cronometer_auth"})
            return
        except CronometerError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "cronometer"})
            return
        finally:
            client.close()

        self._json(200, {"ok": True, "logged": logged, "failed": failed})

    def _handle_undo(self) -> None:
        """DELETE /api?entry_id=X&date=Y — the toast's "undo" tap."""
        if not is_authorized(self.headers):
            self._json(401, {"ok": False, "error": "Unauthorized"})
            return

        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        entry_id = params.get("entry_id")
        if not entry_id or not entry_id.isdigit():
            self._json(400, {"ok": False, "error": "Missing or invalid 'entry_id'"})
            return

        client = CronometerClient()
        try:
            result = client.remove_entries([int(entry_id)], date=params.get("date"))
        except CronometerAuthError as e:
            self._json(401, {"ok": False, "error": str(e), "stage": "cronometer_auth"})
            return
        except CronometerError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "cronometer"})
            return
        finally:
            client.close()

        if result["count"] == 0:
            self._json(404, {"ok": False, "error": "That entry was already gone."})
            return
        self._json(200, {"ok": True, **result})

    def _handle_swap(self) -> None:
        """POST /api?action=swap — remove one entry, log another in its
        place. Body: {entry_id, date, food_id, measure_id, grams, meal}."""
        if not is_authorized(self.headers):
            self._json(401, {"ok": False, "error": "Unauthorized"})
            return

        length = int(self.headers.get("content-length") or 0)
        if length <= 0 or length > 8192:
            self._json(400, {"ok": False, "error": "Missing or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._json(400, {"ok": False, "error": "Body was not valid JSON"})
            return

        try:
            old_entry_id = int(body["entry_id"])
            food_id = int(body["food_id"])
            measure_id = int(body["measure_id"])
            grams = float(body["grams"])
        except (KeyError, TypeError, ValueError):
            self._json(400, {"ok": False,
                             "error": "Need entry_id, food_id, measure_id, grams"})
            return
        date = body.get("date")
        meal = body.get("meal") or "auto"

        client = CronometerClient()
        try:
            # Remove first: if the swap target somehow fails, better to have
            # cleared the wrong entry than to end up with both logged.
            client.remove_entries([old_entry_id], date=date)
            added = client.add_entry(food_id, measure_id, grams, date=date,
                                     diary_group=meal, skip_if_duplicate=False)
        except CronometerAuthError as e:
            self._json(401, {"ok": False, "error": str(e), "stage": "cronometer_auth"})
            return
        except CronometerError as e:
            self._json(502, {"ok": False, "error": str(e), "stage": "cronometer"})
            return
        finally:
            client.close()

        self._json(200, {"ok": True, "removed_entry_id": old_entry_id, "entry": added})

    def _json(self, status: int, payload: dict) -> None:
        blob = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, fmt, *args) -> None:
        """Send access logs to stderr so they land in Vercel's log stream."""
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if v is not None else False
