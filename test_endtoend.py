"""
End-to-end check with both upstreams stubbed: Gemini and Cronometer.

Exercises the whole path a photo takes — vision, matching, logging, summary —
and then the HTTP handler on top of it, including auth. No network, no
credentials, nothing written to a real diary.

    python test_endtoend.py
"""

import io
import json
import os
import sys

os.environ["CRONOMETER_EMAIL"] = "test@example.com"
os.environ["CRONOMETER_PASSWORD"] = "hunter2"
os.environ["CRONOMETER_TIMEZONE"] = "America/Toronto"
os.environ["GEMINI_API_KEY"] = "fake-key"
os.environ["CRONO_VISION_TOKEN"] = "s3cret-token"

import httpx

import pipeline
import vision
from cronometer_client import (CronometerAuthError, CronometerClient, group_name,
                               resolve_diary_group)

_failures = []


def check(label, got, want):
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n       got:  {got!r}\n       want: {want!r}")
        _failures.append(label)


# ── stub Gemini ─────────────────────────────────────────────────────
GEMINI_REPLY = {
    "items": [
        {"label": "Grilled chicken breast", "query": "grilled chicken breast",
         "grams": 185, "confidence": 0.92, "branded": False, "notes": ""},
        {"label": "White rice", "query": "white rice cooked",
         "grams": 210, "confidence": 0.88, "branded": False, "notes": ""},
        {"label": "Mystery sauce", "query": "unidentifiable sauce",
         "grams": 30, "confidence": 0.2, "branded": False, "notes": "too dark to tell"},
    ],
    "meal": "dinner",
    "notes": "",
}
gemini_requests = []


def fake_analyze(image, **kwargs):
    gemini_requests.append(kwargs)
    return vision._parse_response(
        {"candidates": [{"content": {"parts": [{"text": json.dumps(GEMINI_REPLY)}]}}]}
    )


pipeline.vision.analyze_photo = fake_analyze

# ── stub Cronometer ─────────────────────────────────────────────────
FOODS = {
    "grilled chicken breast": [
        {"id": 100, "name": "Chicken, Breast, Grilled", "source": "USDA",
         "measureId": 1000, "measureDisplayName": "100 g", "score": 95.0},
        {"id": 101, "name": "Chicken Nuggets, Breaded", "source": "CRDB",
         "measureId": 1010, "measureDisplayName": "1 piece", "score": 60.0},
    ],
    "white rice cooked": [
        {"id": 200, "name": "Rice, White, Cooked", "source": "USDA",
         "measureId": 2000, "measureDisplayName": "1 cup", "score": 92.0},
    ],
    # A weak, plausible hit — exists so always_log_uncertain has something
    # to force-log. Confidence gating (guess.confidence=0.2, below
    # MIN_VISION_CONFIDENCE) still holds this for review by default
    # regardless of whether a match exists; that's the point of the test
    # at line ~117 below, which stays true whether or not this has a hit.
    "unidentifiable sauce": [
        {"id": 300, "name": "Hot Sauce, Generic", "source": "USDA",
         "measureId": 3000, "measureDisplayName": "1 tbsp", "score": 40.0},
    ],
}
added = []


def crono_stub(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    body = json.loads(request.content) if request.content else {}

    if path.endswith("/login"):
        return httpx.Response(200, json={"id": 4242, "sessionKey": "sess"})
    if path.endswith("/find_food"):
        return httpx.Response(200, json={"foods": FOODS.get(body.get("query"), [])})
    if path.endswith("/get_diary"):
        return httpx.Response(200, json={
            "diary": [{"type": "Serving", "servingId": 900 + i, "foodId": s["foodId"],
                       "foodName": "logged", "measureId": s["measureId"],
                       "grams": s["grams"], "diaryGroup": s["diaryGroup"]}
                      for i, s in enumerate(added)],
            "summary": {"consumed": {"total": 780, "protein_g": 62, "carbs_g": 70, "fat_g": 18},
                        "macros": {"energy": 2200, "protein": 160, "carbs": 200, "fat": 70}},
        })
    if path.endswith("/add_serving"):
        added.append(body["serving"])
        return httpx.Response(200, json={"servingId": 900 + len(added) - 1})
    if path.endswith("/get_nutrients"):
        return httpx.Response(200, json={})
    return httpx.Response(200, json={})


def fresh_client():
    return CronometerClient(http=httpx.Client(transport=httpx.MockTransport(crono_stub)),
                            use_session_cache=False)


# ── dry run ─────────────────────────────────────────────────────────
print("\ndry run")
res = pipeline.log_photo(b"\xff\xd8\xfffake-jpeg", date="2026-08-28", meal="lunch",
                         hint="rice is about 200g", dry_run=True, client=fresh_client())
check("nothing written", added, [])
check("two confident items would log", len(res["logged"]), 2)
check("low-confidence sauce held back", len(res["needs_review"]), 1)
check("hint forwarded to vision", gemini_requests[0]["hint"], "rice is about 200g")
check("explicit meal beats vision's guess", res["meal"], "lunch")
check("would_log carries the resolved food", res["logged"][0]["would_log"]["food_id"], 100)
check("generic entry beat the branded one",
      res["logged"][0]["match"]["best"]["name"], "Chicken, Breast, Grilled")
print(f"       summary: {res['summary']}")

# ── real run ────────────────────────────────────────────────────────
print("\nlogging for real")
added.clear()
res = pipeline.log_photo(b"\xff\xd8\xfffake-jpeg", date="2026-08-28",
                         client=fresh_client())
check("two servings written", len(added), 2)
# The writes go out in parallel now, so which one hits the stub first is
# not ours to predict — look the chicken up by food id rather than by
# position. The *result* order is still stable (it follows the photo), and
# the checks on res[...] below rely on that.
chicken = next(s for s in added if s["foodId"] == 100)
check("grams passed through", chicken["grams"], 185.0)
check("measure id passed through", chicken["measureId"], 1000)
check("date passed through", chicken["day"], "2026-08-28")
# Meal is clock-time only now (see pipeline.py) — Gemini's own read of the
# food is deliberately ignored here, so this has to match whatever the
# clock says right now rather than a fixed fixture value like "dinner".
now_meal = group_name(resolve_diary_group("auto"))
check("no explicit meal -> pure clock time", res["meal"], now_meal)
check("diary group on the serving matches that clock bucket",
      chicken["diaryGroup"], resolve_diary_group("auto"))

# The web page always sends meal="auto" explicitly (never omits it), so
# that literal string has to behave exactly like not passing meal at all.
added.clear()
res = pipeline.log_photo(b"\xff\xd8\xfffake-jpeg", date="2026-08-28", meal="auto",
                         client=fresh_client())
check("meal='auto' behaves like no meal at all", res["meal"], now_meal)
check("order packs the group", added[0]["order"], (resolve_diary_group("auto") << 16) | 1)
check("results stay in photo order, whatever order the writes landed in",
      [e["match"]["best"]["food_id"] for e in res["logged"]], [100, 200])
check("entry id returned", res["logged"][0]["entry"]["entry_id"] in (900, 901), True)
check("daily totals attached", res["daily"]["consumed"]["calories"], 780.0)
check("remaining computed", res["daily"]["remaining"]["calories"], 1420.0)
print(f"       summary: {res['summary']}")

# ── nothing in the photo ────────────────────────────────────────────
print("\nempty photo")
GEMINI_REPLY_EMPTY = {"items": [], "meal": "unknown", "notes": "no food visible"}
pipeline.vision.analyze_photo = lambda image, **kw: vision._parse_response(
    {"candidates": [{"content": {"parts": [{"text": json.dumps(GEMINI_REPLY_EMPTY)}]}}]})
added.clear()
res = pipeline.log_photo(b"\xff\xd8\xff", client=fresh_client())
check("no writes attempted", added, [])
check("says so plainly", res["summary"], "No food found in that photo.")
pipeline.vision.analyze_photo = fake_analyze

# ── one food blows up, the rest still log ───────────────────────────
print("\npartial failure")
added.clear()
calls = {"n": 0}
original_search = CronometerClient.search_foods


def flaky_search(self, query, limit=15):
    if query == "white rice cooked":
        raise RuntimeError("upstream hiccup")
    return original_search(self, query, limit=limit)


CronometerClient.search_foods = flaky_search
res = pipeline.log_photo(b"\xff\xd8\xff", date="2026-08-28", client=fresh_client())
CronometerClient.search_foods = original_search
check("the healthy item still logged", len(res["logged"]), 1)
check("the broken one is reported, not swallowed", len(res["failed"]), 1)
check("error text preserved", "upstream hiccup" in res["failed"][0]["error"], True)

# ── HTTP handler ────────────────────────────────────────────────────
print("\nhttp handler")
sys.path.insert(0, "api")
import index as api_log  # noqa: E402

api_log.log_photo = lambda image, **kw: {
    "date": "2026-08-28", "meal": "lunch", "dry_run": kw.get("dry_run", False),
    "logged": [], "needs_review": [], "failed": [], "daily": None,
    "vision": {}, "summary": "ok", "_received_bytes": len(image), "_kwargs": {
        k: v for k, v in kw.items()},
}


class FakeHandler(api_log.handler):
    """Drives the handler without a socket."""

    def __init__(self, method, path, headers, body=b""):
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.headers = headers
        self.path = path
        self.command = method
        self.request_version = "HTTP/1.1"
        self.client_address = ("127.0.0.1", 0)
        self.status = None

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, name, value):
        self.sent_headers = getattr(self, "sent_headers", {})
        self.sent_headers[name.lower()] = value

    def end_headers(self):
        pass

    def address_string(self):
        return "test"


def call(method, path, headers, body=b""):
    h = FakeHandler(method, path, headers, body)
    getattr(h, f"do_{method}")()
    return h.status, json.loads(h.wfile.getvalue() or b"{}")


AUTH = {"authorization": "Bearer s3cret-token", "content-length": "5", "content-type": "image/jpeg"}

status, out = call("POST", "/api", {**AUTH, "authorization": "Bearer wrong"}, b"12345")
check("wrong token rejected", status, 401)

status, out = call("POST", "/api", {**AUTH, "authorization": ""}, b"12345")
check("missing token rejected", status, 401)

status, out = call("POST", "/api", {**AUTH, "content-length": "0"}, b"")
check("empty body rejected", status, 400)

status, out = call("POST", "/api?meal=dinner&dry_run=true", AUTH, b"12345")
check("raw image body accepted", status, 200)
check("bytes reached the pipeline", out["_received_bytes"], 5)
check("query params parsed", out["_kwargs"]["meal"], "dinner")
check("dry_run coerced to bool", out["_kwargs"]["dry_run"], True)
check("summary hoisted to the top level", out["summary"], "ok")

payload = json.dumps({"image": "aGVsbG8=", "meal": "breakfast", "hint": "oatmeal"}).encode()
status, out = call("POST", "/api",
                   {"authorization": "Bearer s3cret-token",
                    "content-length": str(len(payload)),
                    "content-type": "application/json"}, payload)
check("json body accepted", status, 200)
check("base64 decoded", out["_received_bytes"], 5)
check("json fields become params", out["_kwargs"]["hint"], "oatmeal")

bad = b'{"image": "not!!base64!!"}'
status, out = call("POST", "/api",
                   {"authorization": "Bearer s3cret-token",
                    "content-length": str(len(bad)),
                    "content-type": "application/json"}, bad)
check("garbage base64 gives a 4xx not a 500", status in (400,), True)

status, out = call("POST", "/api", {**AUTH, "content-length": str(20 * 1024 * 1024)}, b"")
check("oversized image rejected", status, 413)

status, out = call("GET", "/api", {})
check("health check needs no auth", status, 200)
check("health check reports config", out["configured"]["gemini"], True)

saved = os.environ.pop("CRONO_VISION_TOKEN")
status, out = call("POST", "/api", AUTH, b"12345")
check("unset token refuses everything", status, 401)
os.environ["CRONO_VISION_TOKEN"] = saved

# ── cookie login (/api/auth) ────────────────────────────────────────
print("\ncookie login")
import auth as api_auth  # noqa: E402
from webauth import COOKIE_NAME, is_authorized  # noqa: E402


def call_auth(method, path, headers, body=b""):
    h = FakeHandler(method, path, headers, body)
    h.__class__ = type("FakeAuthHandler", (api_auth.handler,), {
        "send_response": FakeHandler.send_response,
        "send_header": FakeHandler.send_header,
        "end_headers": lambda self: None,
        "address_string": lambda self: "test",
    })
    getattr(h, f"do_{method}")()
    return h.status, json.loads(h.wfile.getvalue() or b"{}"), getattr(h, "sent_headers", {})


status, out, headers = call_auth("GET", "/api/auth", {})
check("not authed with no cookie", out["authed"], False)

body = json.dumps({"token": "wrong-token"}).encode()
status, out, headers = call_auth("POST", "/api/auth",
                                  {"content-length": str(len(body))}, body)
check("wrong token rejected", status, 401)
check("no cookie set on rejection", "set-cookie" in headers, False)

body = json.dumps({"token": "s3cret-token"}).encode()
status, out, headers = call_auth("POST", "/api/auth",
                                  {"content-length": str(len(body))}, body)
check("right token accepted", status, 200)
check("authed true on success", out["authed"], True)
check("cookie is HttpOnly + Secure + SameSite=Strict",
      all(f in headers["set-cookie"] for f in ("HttpOnly", "Secure", "SameSite=Strict")), True)

set_cookie_value = headers["set-cookie"].split(";")[0]  # "cv_auth=<token>"
status, out, headers = call_auth("GET", "/api/auth", {"cookie": set_cookie_value})
check("cookie alone authenticates", out["authed"], True)

status, out, headers = call_auth("POST", "/api/auth?logout=1", {}, b"")
check("logout clears the cookie", "Max-Age=0" in headers["set-cookie"], True)

check("is_authorized accepts bearer",
      is_authorized({"authorization": "Bearer s3cret-token"}), True)
check("is_authorized accepts the cookie",
      is_authorized({"cookie": f"{COOKIE_NAME}=s3cret-token"}), True)
check("is_authorized rejects a wrong cookie value",
      is_authorized({"cookie": f"{COOKIE_NAME}=nope"}), False)
check("is_authorized rejects an unrelated cookie",
      is_authorized({"cookie": "other=s3cret-token"}), False)

# ── undo / swap (/api DELETE, /api?action=swap) ─────────────────────
print("\nundo and swap")

# _handle_undo/_handle_swap each construct their own CronometerClient()
# with no args — correct in production, but that means a real one with a
# real httpx.Client would try to hit the actual network here. Point the
# name the handler module resolves at call time to our stub instead.
api_log.CronometerClient = lambda *a, **kw: fresh_client()

added.clear()
res = pipeline.log_photo(b"\xff\xd8\xfffake-jpeg", date="2026-08-28", client=fresh_client())
logged_entry_id = res["logged"][0]["entry"]["entry_id"]

status, out = call("DELETE", f"/api?entry_id={logged_entry_id}&date=2026-08-28", AUTH, b"")
check("undo removes the entry", status, 200)
check("undo reports the removed id", out["removed"], [logged_entry_id])

status, out = call("DELETE", "/api?entry_id=999999&date=2026-08-28", AUTH, b"")
check("undoing something already gone is a 404, not a 500", status, 404)

status, out = call("DELETE", "/api?date=2026-08-28", AUTH, b"")
check("undo without entry_id is a 400", status, 400)

status, out = call("DELETE", f"/api?entry_id={logged_entry_id}",
                   {**AUTH, "authorization": "Bearer wrong"}, b"")
check("undo respects auth too", status, 401)

swap_body = json.dumps({
    "entry_id": 900, "date": "2026-08-28",
    "food_id": 200, "measure_id": 2000, "grams": 150, "meal": "dinner",
}).encode()
status, out = call("POST", "/api?action=swap",
                   {**AUTH, "content-length": str(len(swap_body)),
                    "content-type": "application/json"}, swap_body)
check("swap succeeds", status, 200)
check("swap reports the new entry's food", out["entry"]["food_id"], 200)
check("swap reports what it removed", out["removed_entry_id"], 900)

bad_swap = json.dumps({"entry_id": 900}).encode()  # missing food_id etc.
status, out = call("POST", "/api?action=swap",
                   {**AUTH, "content-length": str(len(bad_swap)),
                    "content-type": "application/json"}, bad_swap)
check("swap validates its body", status, 400)

# ── always_log_uncertain (the camera page's mode) ───────────────────
print("\nalways_log_uncertain")
added.clear()
res = pipeline.log_photo(b"\xff\xd8\xfffake-jpeg", date="2026-08-28",
                         always_log_uncertain=True, client=fresh_client())
check("all three items attempted, none held for review", len(res["needs_review"]), 0)
check("the two clean matches logged unflagged", len(res["logged"]), 3)
sauce_entry = next(e for e in res["logged"] if e["guess"]["label"] == "Mystery sauce")
check("the low-confidence one is logged but flagged", sauce_entry.get("flagged"), True)
check("it carries the reason it was flagged for", "unsure this is" in sauce_entry["reason"], True)
clean_entry = next(e for e in res["logged"] if e["guess"]["label"] == "Grilled chicken breast")
check("clean matches aren't flagged", clean_entry.get("flagged", False), False)
check("summary calls out the unsure count", "unsure" in res["summary"], True)

status, out = call("POST", "/api?always_log=true",
                   {**AUTH, "content-length": "5", "content-type": "image/jpeg"}, b"12345")
check("always_log query param reaches the pipeline", out["_kwargs"].get("always_log_uncertain"), True)

status, out = call("POST", "/api",
                   {**AUTH, "content-length": "5", "content-type": "image/jpeg"}, b"12345")
check("default (Shortcut path) stays conservative",
      out["_kwargs"].get("always_log_uncertain"), False)

# ── daily totals GET (/api?daily=1) ──────────────────────────────────
print("\ndaily totals")
api_log.CronometerClient = lambda *a, **kw: fresh_client()

status, out = call("GET", "/api?daily=1", AUTH)
check("authenticated daily request succeeds", status, 200)
check("daily payload present", "daily" in out, True)
check("daily doesn't leak the full entry list", "entries" in out.get("daily", {}), False)

status, out = call("GET", "/api?daily=1", {"authorization": "Bearer wrong"})
check("unauthenticated daily request falls back to the health check",
      "daily" not in out and out.get("service") == "crono-vision", True)

status, out = call("GET", "/api", {})
check("plain GET with no daily param is still just the health check",
      "daily" not in out, True)

# ── warm-up (/api?warm=1) ────────────────────────────────────────────
# Fired when the camera comes up, so the cold start and the Cronometer
# login are paid before someone presses the shutter rather than by them.
print("\nwarm-up")
status, out = call("GET", "/api?warm=1", AUTH)
check("warm-up succeeds", (status, out.get("warm")), (200, True))
check("warm-up says nothing about the diary", "daily" in out, False)

status, out = call("GET", "/api?warm=1", {"authorization": "Bearer wrong"})
check("unauthenticated warm-up falls back to the health check",
      out.get("service"), "crono-vision")


def failing_login(*a, **kw):
    c = fresh_client()
    c.login = lambda: (_ for _ in ()).throw(CronometerAuthError("nope"))
    c.ensure_auth = c.login
    return c


api_log.CronometerClient = failing_login
status, out = call("GET", "/api?warm=1", AUTH)
check("a failed warm-up is reported, not fatal — it's only an optimization",
      (status, out.get("warm")), (200, False))
api_log.CronometerClient = lambda *a, **kw: fresh_client()

# ── two-phase: analyze writes nothing, commit writes ─────────────────
# The camera page takes these as separate requests so it can draw its
# detection markers without waiting on the diary. The contract that makes
# that safe is the one checked here: analyze must not touch the diary, and
# commit must be replayable without doubling anything up.
print("\ntwo-phase analyze/commit")
added.clear()
found = pipeline.analyze(b"\xff\xd8\xfffake-jpeg", date="2026-08-28", meal="lunch",
                         client=fresh_client())
check("analyze wrote nothing", added, [])
check("two items pending", len(found["pending"]), 2)
check("the unsure one is still held back", len(found["needs_review"]), 1)
check("each pending item carries a plan",
      all("plan" in e for e in found["pending"]), True)
check("the plan is what add_entry needs",
      sorted(found["pending"][0]["plan"]), ["food_id", "food_name", "grams", "measure_id"])
check("markers survive the split — boxes still on the guesses",
      all("box_2d" in e["guess"] for e in found["pending"]), True)

client = fresh_client()
logged, failed = pipeline.commit(found["pending"], date="2026-08-28",
                                 meal="lunch", client=client)
check("commit wrote both", len(added), 2)
check("commit reported both", len(logged), 2)
check("nothing failed", failed, [])
check("entry ids came back", all(e["entry"]["entry_id"] is not None for e in logged), True)
check("results follow photo order, not write order",
      [e["match"]["best"]["food_id"] for e in logged], [100, 200])

# The point of splitting: a retry replays identical plans, so the duplicate
# check actually matches. Re-running the whole pipeline wouldn't — a fresh
# portion estimate of 148g against a logged 150g slips straight past it.
logged2, _ = pipeline.commit(found["pending"], date="2026-08-28",
                             meal="lunch", client=fresh_client())
check("replaying a commit doesn't double-log", len(added), 2)
check("the replay reports the entries that already existed",
      all(e["entry"].get("skipped") for e in logged2), True)

# One shared diary snapshot for the batch, not one read per item.
reads = {"n": 0}
original_get_diary = CronometerClient.get_diary


def counting_get_diary(self, date=None):
    reads["n"] += 1
    return original_get_diary(self, date)


CronometerClient.get_diary = counting_get_diary
added.clear()
pipeline.commit(found["pending"], date="2026-08-28", meal="lunch", client=fresh_client())
CronometerClient.get_diary = original_get_diary
check("the day is read once for the whole batch, not once per item", reads["n"], 1)

# ── commit endpoint ──────────────────────────────────────────────────
print("\ncommit endpoint")
api_log.CronometerClient = lambda *a, **kw: fresh_client()
added.clear()


def commit_call(body, headers=None):
    blob = json.dumps(body).encode()
    return call("POST", "/api?action=commit",
                {"authorization": "Bearer s3cret-token",
                 "content-length": str(len(blob)),
                 "content-type": "application/json", **(headers or {})}, blob)

status, out = commit_call({"date": "2026-08-28", "meal": "lunch",
                           "items": found["pending"]})
check("commit endpoint succeeds", status, 200)
check("it logged both items", len(out["logged"]), 2)
check("entries carry ids the page's Reject button needs",
      all(e["entry"]["entry_id"] is not None for e in out["logged"]), True)
check("the page's own guess/match data comes back with it",
      "match" in out["logged"][0] and "guess" in out["logged"][0], True)

status, out = commit_call({"items": []})
check("an empty batch is a no-op, not an error", (status, out["logged"]), (200, []))

status, out = commit_call({"items": [{"plan": {"food_id": 100}}]})
check("an incomplete plan is a 400", status, 400)

status, out = commit_call({"items": [{"plan": {"food_id": 100, "measure_id": 1000,
                                               "grams": 0}}]})
check("zero grams is a 400", status, 400)

status, out = commit_call({"nope": 1})
check("a body with no items list is a 400", status, 400)

status, out = commit_call({"items": found["pending"]},
                          {"authorization": "Bearer wrong"})
check("commit respects auth too", status, 401)

print()
if _failures:
    print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
    sys.exit(1)
print("all end-to-end checks passed")
