"""
Offline checks — no network, no credentials.

Covers the parts that are easy to get subtly wrong: the ranking maths, the
timezone-sensitive date handling, and the response parsers. The network paths
are exercised against a stubbed httpx transport rather than the real API, so
this suite is safe to run in CI and safe to run repeatedly.

    python test_offline.py
"""

import json
import os
import sys

os.environ.setdefault("CRONOMETER_EMAIL", "test@example.com")
os.environ.setdefault("CRONOMETER_PASSWORD", "hunter2")
os.environ.setdefault("CRONOMETER_TIMEZONE", "America/Toronto")

import httpx

import matcher
import vision
from cronometer_client import (
    CronometerClient, FoodHit, DIARY_GROUPS,
    normalize_date, resolve_diary_group, today, _extract_entry_id,
)

_failures = []


def check(label, got, want):
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n       got:  {got!r}\n       want: {want!r}")
        _failures.append(label)


def check_true(label, cond, detail=""):
    check(label + (f" ({detail})" if detail and not cond else ""), bool(cond), True)


# ── dates & meals ───────────────────────────────────────────────────
print("\ndates and diary groups")
check("normalize_date passthrough", normalize_date("2026-08-28"), "2026-08-28")
check("normalize_date today", normalize_date(None), today())
check("normalize_date 'today'", normalize_date("today"), today())
check_true("normalize_date 'yesterday' differs", normalize_date("yesterday") != today())
try:
    normalize_date("28/08/2026")
    check("normalize_date rejects junk", "no raise", "ValueError")
except ValueError:
    check("normalize_date rejects junk", "ValueError", "ValueError")

check("explicit meal", resolve_diary_group("dinner"), DIARY_GROUPS["dinner"])
check("case insensitive", resolve_diary_group("BREAKFAST"), DIARY_GROUPS["breakfast"])
from datetime import datetime
check("auto @ 08:00 -> breakfast", resolve_diary_group("auto", datetime(2026, 8, 28, 8, 0)), 1)
check("auto @ 12:30 -> lunch", resolve_diary_group("auto", datetime(2026, 8, 28, 12, 30)), 2)
check("auto @ 19:00 -> dinner", resolve_diary_group("auto", datetime(2026, 8, 28, 19, 0)), 3)
check("auto @ 22:30 -> snacks", resolve_diary_group("auto", datetime(2026, 8, 28, 22, 30)), 4)
check("None -> auto", resolve_diary_group(None, datetime(2026, 8, 28, 12, 0)), 2)

# ── matching ────────────────────────────────────────────────────────
print("\ntoken f1")
check_true("exact beats superset",
           matcher.token_f1("chicken breast", "Chicken Breast")
           > matcher.token_f1("chicken breast", "Chicken Breast Nuggets, Breaded, Frozen"))
check_true("unrelated scores zero",
           matcher.token_f1("chicken breast", "Banana, raw") == 0.0)
check_true("stopwords ignored",
           matcher.token_f1("chicken with rice", "Chicken and Rice") > 0.9)

print("\nranking")
hits = [
    FoodHit(1, "Chicken Breast Nuggets, Breaded, Frozen", "CRDB", 11, "1 piece", 95.0),
    FoodHit(2, "Chicken, Breast, Grilled", "USDA", 22, "100 g", 88.0),
    FoodHit(3, "Banana, Raw", "USDA", 33, "1 medium", 40.0),
]
result = matcher.match("grilled chicken breast", hits)
check("generic grilled entry wins over branded nuggets", result.best.hit.food_id, 2)
check_true("banana ranks last",
           matcher.score_hits("grilled chicken breast", hits)[-1].hit.food_id == 3)
check_true("clean match is not flagged", not result.needs_review, result.reason)

branded = matcher.match("chicken nuggets", hits, prefer_branded=True)
check("prefer_branded flips the source bonus", branded.best.hit.food_id, 1)

no_measure = matcher.match("chicken breast", [FoodHit(9, "Chicken Breast", "USDA", None, "", 99.0)])
check_true("hit without a measure is flagged", no_measure.needs_review, no_measure.reason)
check("empty results flagged", matcher.match("x", []).needs_review, True)

# Two foods that are genuinely indistinguishable from the query alone:
# same tokens matched, same source, same Cronometer score.
ambiguous = matcher.match("plain yogurt", [
    FoodHit(1, "Yogurt, Plain, Whole Milk", "USDA", 1, "1 cup", 90.0),
    FoodHit(2, "Yogurt, Plain, Low Fat", "USDA", 2, "1 cup", 90.0),
])
check_true("near-ties are flagged for review", ambiguous.needs_review, ambiguous.reason)

# Known gap: token overlap is spelling-exact, so "yoghurt" shares nothing
# with "yogurt". Cronometer's own score is what rescues these, which is a
# large part of why it keeps 40% of the weight.
check("spelling variants score 0 on tokens", matcher.token_f1("yogurt", "Yoghurt"), 0.0)

zero_scores = matcher.match("oat milk", [
    FoodHit(1, "Oat Milk", "USDA", 1, "1 cup", 0.0),
    FoodHit(2, "Whole Milk", "USDA", 2, "1 cup", 0.0),
])
check("falls back gracefully when every score is 0", zero_scores.best.hit.food_id, 1)

# ── vision parsing ──────────────────────────────────────────────────
print("\nvision")
check("jpeg sniffed", vision.sniff_mime(b"\xff\xd8\xff\xe0rest"), "image/jpeg")
check("png sniffed", vision.sniff_mime(b"\x89PNG\r\n\x1a\nrest"), "image/png")
check("heic sniffed", vision.sniff_mime(b"\x00\x00\x00\x18ftypheic...."), "image/heic")

parsed = vision._parse_response({"candidates": [{"content": {"parts": [{"text": json.dumps({
    "items": [
        {"label": "Grilled chicken", "query": "grilled chicken breast",
         "grams": 180, "confidence": 0.9, "branded": False, "notes": "",
         "box_2d": [120, 300, 640, 810]},
        {"label": "Side salad", "query": "", "grams": 50, "confidence": 0.4, "branded": False},
        {"label": "zero grams", "query": "rice", "grams": 0, "confidence": 0.4, "branded": False},
        {"label": "", "query": "", "grams": 20, "confidence": 0.4, "branded": False},
        {"label": "no grams", "query": "bread", "confidence": 0.4, "branded": False},
    ],
    "meal": "dinner", "notes": "plate looks large",
})}]}}]})
check("rows without a usable amount or name are dropped", len(parsed.items), 2)
check("query falls back to label", parsed.items[1].query, "Side salad")
check("grams parsed", parsed.items[0].grams, 180.0)
check("meal parsed", parsed.meal, "dinner")
check("box_2d parsed when present", parsed.items[0].box_2d, (120, 300, 640, 810))
check("box_2d absent when not returned", parsed.items[1].box_2d, None)

for label, box, want in [
    ("valid box", [10, 20, 500, 600], (10, 20, 500, 600)),
    ("floats get rounded", [10.6, 20.2, 500.4, 600.9], (11, 20, 500, 601)),
    ("out-of-range gets clamped", [-50, 20, 1500, 600], (0, 20, 1000, 600)),
    ("wrong length", [10, 20, 500], None),
    ("not a list", "nope", None),
    ("zero-height box is degenerate", [100, 20, 100, 600], None),
    ("inverted box is degenerate", [500, 20, 100, 600], None),
    ("non-numeric entries", [10, "x", 500, 600], None),
]:
    check(f"_parse_box: {label}", vision._parse_box(box), want)

for label, payload in [
    ("blocked prompt", {"promptFeedback": {"blockReason": "SAFETY"}}),
    ("empty text", {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}),
    ("not json", {"candidates": [{"content": {"parts": [{"text": "sorry!"}]}}]}),
]:
    try:
        vision._parse_response(payload)
        check(f"raises on {label}", "no raise", "VisionError")
    except vision.VisionError:
        check(f"raises on {label}", "VisionError", "VisionError")

# ── client, against a stub transport ────────────────────────────────
print("\nclient (stubbed transport)")
calls = []


def stub(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    calls.append(path)

    if path.endswith("/login"):
        return httpx.Response(200, json={"id": 4242, "sessionKey": "sess-abc"})
    if path.endswith("/find_food"):
        return httpx.Response(200, json={"foods": [
            {"id": 7, "name": "Chicken, Breast, Grilled", "source": "USDA",
             "measureId": 70, "measureDisplayName": "100 g", "score": 91.5},
            {"id": 8, "name": "Malformed"},  # no id issues, but no measure
        ]})
    if path.endswith("/get_diary"):
        return httpx.Response(200, json={
            "diary": [
                {"type": "Serving", "servingId": 555, "foodId": 7, "foodName": "Chicken",
                 "measureId": 70, "grams": 180.0, "order": (3 << 16) | 1},
                {"type": "Note", "servingId": 999},
            ],
            "summary": {"consumed": {"total": 1450, "protein_g": 110, "carbs_g": 120, "fat_g": 50},
                        "macros": {"energy": 2200, "protein": 160, "carbs": 200, "fat": 70}},
        })
    if path.endswith("/get_nutrients"):
        return httpx.Response(200, json={"nutrients": []})
    if path.endswith("/add_serving"):
        return httpx.Response(200, json={"serving": {"servingId": 777}})
    return httpx.Response(200, json={})


http = httpx.Client(transport=httpx.MockTransport(stub))
client = CronometerClient(http=http, use_session_cache=False)

hits = client.search_foods("grilled chicken")
check("search returns parsed hits", len(hits), 2)
check("hit fields mapped", (hits[0].food_id, hits[0].measure_id, hits[0].score), (7, 70, 91.5))
check("missing measure -> None", hits[1].measure_id, None)
check("logged in lazily", client.user_id, 4242)

entries = client.get_diary("2026-08-28")
check("non-Serving rows filtered", len(entries), 1)
check("diary group unpacked from order", entries[0].diary_group, 3)
check("meal name derived", entries[0].meal, "dinner")

daily = client.get_daily_nutrition("2026-08-28")
check("consumed parsed", daily["consumed"]["calories"], 1450.0)
check("remaining computed", daily["remaining"]["calories"], 750.0)

added = client.add_entry(7, 70, 200.0, date="2026-08-28", diary_group="lunch")
check("entry id extracted from nested payload", added["entry_id"], 777)
check("meal recorded", added["meal"], "lunch")
check_true("not skipped", not added["skipped"])

dupe = client.add_entry(7, 70, 180.0, date="2026-08-28", diary_group="dinner")
check("duplicate detected against live diary", dupe["skipped"], True)
check("duplicate returns existing id", dupe["entry_id"], 555)

try:
    client.add_entry(7, 70, -5)
    check("rejects non-positive grams", "no raise", "ValueError")
except ValueError:
    check("rejects non-positive grams", "ValueError", "ValueError")

check("entry id from bare int", _extract_entry_id(12345), 12345)
check("entry id from flat dict", _extract_entry_id({"id": 5}), 5)
check("entry id absent", _extract_entry_id({"nope": 1}), None)

# session expiry -> transparent re-login
expired = {"n": 0}


def flaky(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/login"):
        return httpx.Response(200, json={"id": 1, "sessionKey": "fresh"})
    expired["n"] += 1
    if expired["n"] == 1:
        return httpx.Response(401)
    return httpx.Response(200, json={"foods": []})


c2 = CronometerClient(http=httpx.Client(transport=httpx.MockTransport(flaky)),
                      use_session_cache=False)
c2._token, c2.user_id = "stale", 1
check("401 triggers re-auth and retry", c2.search_foods("x"), [])

# ── vision model fallback ────────────────────────────────────────────
print("\nvision model fallback")

GOOD_REPLY = json.dumps({
    "items": [{"label": "Toast", "query": "toast", "grams": 30,
               "confidence": 0.9, "branded": False, "notes": ""}],
    "meal": "breakfast", "notes": "",
})
calls_seen = []


def make_gemini_stub(fail_models):
    """fail_models: {model_name: status_code} — that model 503s/429s/etc.
    Anything not listed succeeds."""
    def stub(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        calls_seen.append(model)
        if model in fail_models:
            return httpx.Response(fail_models[model], text="overloaded")
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": GOOD_REPLY}]}}]
        })
    return stub


# Reference the real chain rather than hardcoding names, so this doesn't
# silently stop testing anything real the next time a model gets
# deprecated out from under DEFAULT_FALLBACK_MODELS (as already happened
# once — the original fallback here was gemini-2.5-flash).
FULL_CHAIN = [vision.DEFAULT_MODEL] + list(vision.DEFAULT_FALLBACK_MODELS)

calls_seen.clear()
http1 = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({FULL_CHAIN[0]: 503})))
result = vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http1)
check("falls back to the next model on 503", calls_seen, FULL_CHAIN[:2])
check("fallback call still returns usable items", len(result.items), 1)

calls_seen.clear()
http1b = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({FULL_CHAIN[0]: 404})))
vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http1b)
check("a deprecated/renamed model (404) also triggers fallback", calls_seen, FULL_CHAIN[:2])

calls_seen.clear()
http2 = httpx.Client(transport=httpx.MockTransport(make_gemini_stub({})))
vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http2)
check("primary succeeding never touches the fallback", calls_seen, [FULL_CHAIN[0]])

calls_seen.clear()
http3 = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({FULL_CHAIN[0]: 401})))
try:
    vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http3)
    check("non-retryable status skips the fallback chain", "no raise", "VisionError")
except vision.VisionError:
    check("non-retryable status skips the fallback chain", calls_seen, [FULL_CHAIN[0]])

calls_seen.clear()
http4 = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({m: 503 for m in FULL_CHAIN})))
try:
    vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http4)
    check("every model overloaded still raises", "no raise", "VisionError")
except vision.VisionError as e:
    check("every model overloaded still raises", isinstance(e, vision.VisionOverloaded), True)
    check("tried every candidate before giving up", calls_seen, FULL_CHAIN)

os.environ["GEMINI_FALLBACK_MODELS"] = "gemini-custom-a,gemini-custom-b"
calls_seen.clear()
http5 = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({FULL_CHAIN[0]: 503, "gemini-custom-a": 503})))
vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http5)
check("GEMINI_FALLBACK_MODELS overrides the built-in chain",
      calls_seen, [FULL_CHAIN[0], "gemini-custom-a", "gemini-custom-b"])
del os.environ["GEMINI_FALLBACK_MODELS"]

os.environ["GEMINI_FALLBACK_MODELS"] = ""
calls_seen.clear()
http6 = httpx.Client(transport=httpx.MockTransport(
    make_gemini_stub({FULL_CHAIN[0]: 503})))
try:
    vision.analyze_photo(b"\xff\xd8\xff", api_key="k", http=http6)
    check("empty GEMINI_FALLBACK_MODELS disables fallback", "no raise", "VisionOverloaded")
except vision.VisionOverloaded:
    check("empty GEMINI_FALLBACK_MODELS disables fallback", calls_seen, [FULL_CHAIN[0]])
del os.environ["GEMINI_FALLBACK_MODELS"]

print()
if _failures:
    print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
    sys.exit(1)
print("all offline checks passed")
