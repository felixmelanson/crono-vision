"""
Photo → list of food guesses, via Gemini.

Talks to the Generative Language REST API directly with httpx instead of
pulling in the google-genai SDK. One less dependency to keep pinned, one
less thing to break a serverless bundle, and the request is about twenty
lines. If you'd rather use the SDK, this is the only file that changes.

    guess = analyze_photo(open("lunch.jpg", "rb").read())
    for item in guess.items:
        print(item.query, item.grams)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

log = logging.getLogger("crono_vision.vision")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# A Lite model, deliberately, and this is the single biggest lever on how
# long someone stands there holding their phone. Measured on the same
# photo, Aug 2026:
#
#   gemini-3.5-flash-lite    ~1.8s
#   gemini-3.6-flash         ~9.4s
#   gemini-3.7-flash         ~6-8s, and the first to start returning 429
#
# Same four items, same bounding boxes, same portion estimates — the big
# models spend the extra seconds *thinking* (they report hundreds of
# thinking tokens; the Lite models report none) and arrive at the same
# answer. Identifying an apple is not a reasoning problem. Naming what's
# on a plate is the one thing every one of these models is already good
# at, so buying more capability here buys nothing but latency.
#
# Override with GEMINI_MODEL if a plate ever genuinely stumps this one.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Tried in order after the primary model, on either a capacity failure (the
# model is overloaded or rate-limited) or a 404 (the model id was
# deprecated or renamed out from under us — this list has already gone
# stale once, from a pinned gemini-2.5-flash Google cut off from new
# callers). gemini-flash-lite-latest is deliberately an alias rather than a
# version: it's Google's own pointer to whatever Lite model is current, so
# it can't go stale the way a pinned name can.
#
# Ordered fastest-first, which is the opposite of the obvious instinct. A
# fallback fires when something is already wrong, i.e. exactly when the
# person has already been waiting — that is the worst moment to reach for
# the slowest model in the list. gemini-3.6-flash is last because it's the
# only genuinely different one, worth ~9s only when everything quick has
# already refused.
#
# Override with GEMINI_FALLBACK_MODELS="model-a,model-b" (empty string
# disables fallback entirely).
DEFAULT_FALLBACK_MODELS = ("gemini-3.1-flash-lite", "gemini-flash-lite-latest",
                           "gemini-3.6-flash")

PROMPT = """You are a nutrition-logging assistant. Identify every distinct food \
and drink in this photo and estimate how much of each is present.

For each item:
- `label`: what a person would call it, e.g. "grilled chicken breast".
- `query`: a search string for a nutrition database. Keep it generic and \
short (2-4 words). Include the cooking method when it changes the nutrition \
("grilled", "fried", "raw"). Do NOT include brand names unless packaging is \
clearly visible in the photo, and do not include amounts.
- `grams`: estimated edible weight in grams. If a hand is visible in the \
frame, prefer it as your scale reference over anything else — plate and \
bowl sizes vary a lot, but an adult palm (excluding fingers) is reliably \
about 8-9cm wide and a full hand tip-to-wrist is about 18cm, so it pins \
down scale much more tightly. Otherwise use whatever else is visible: a \
dinner plate ~27cm across, a fork ~19cm, a standard soda can 330ml, a \
slice of sandwich bread ~30g, a chicken breast ~150-200g. Estimate the \
food only, not the plate, packaging, or hand. For drinks, use millilitres \
as grams.
- `confidence`: 0.0-1.0, how sure you are this is what the food is.
- `branded`: true only if a brand name or product package is legible.
- `box_2d`: a tight bounding box around this item, as [ymin, xmin, ymax, \
xmax] with each value an integer 0-1000 normalized to the image's height \
and width. Box the food itself, not its container or shadow.

Also return `course`: "meal" if this is a proper meal, "snack" if it's a \
light bite — a piece of fruit, a handful of nuts, a single drink, a small \
bowl of one thing. Judge the food and the amount, not the hour. Say \
"unknown" if it's genuinely borderline.

Return only these fields. Do not explain, caption, or add commentary: every \
token you spend on prose is a second the person holding the phone spends \
staring at a frozen frame.

Combine things that would be logged as one food (a sandwich's bread, a salad's \
dressing) only when they cannot be separated visually. Skip garnishes under \
about 5 grams. If the image contains no food at all, return an empty items list."""

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "query": {"type": "STRING"},
                    "grams": {"type": "NUMBER"},
                    "confidence": {"type": "NUMBER"},
                    "branded": {"type": "BOOLEAN"},
                    "box_2d": {
                        "type": "ARRAY", "items": {"type": "INTEGER"},
                        "minItems": 4, "maxItems": 4,
                    },
                },
                "required": ["label", "query", "grams", "confidence", "branded"],
                "propertyOrdering": ["label", "query", "grams", "confidence", "branded", "box_2d"],
            },
        },
        # Deliberately NOT breakfast/lunch/dinner. Naming the time slot is
        # the authority this model had once and misused: a steak
        # photographed at 12:27am came back "lunch" because the plate looked
        # like lunch. It has no vocabulary for a slot now, so the clock owns
        # that and this only answers meal-or-snack — a judgment about the
        # food itself, which is the part it's actually good at.
        "course": {"type": "STRING", "enum": ["meal", "snack", "unknown"]},
    },
    "required": ["items"],
    "propertyOrdering": ["items", "course"],
}


class VisionError(Exception):
    """Gemini didn't give us something usable."""


class VisionOverloaded(VisionError):
    """The model said no capacity, not "your request was bad" — worth
    trying a different model rather than giving up."""


@dataclass
class FoodGuess:
    label: str
    query: str
    grams: float
    confidence: float
    branded: bool = False
    # No longer requested — see the schema. Kept as a tolerated field so a
    # model that volunteers one anyway still parses, and so the response
    # shape doesn't change for anything already reading it.
    notes: str = ""
    # [ymin, xmin, ymax, xmax], each 0-1000 normalized to image height/width
    # (Gemini's standard object-detection convention) — None when the model
    # didn't return one. Lets the capture UI draw a marker over the actual
    # detected region instead of a generic "processing" spinner.
    box_2d: Optional[tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhotoAnalysis:
    items: list[FoodGuess] = field(default_factory=list)
    # "meal" | "snack" | "unknown" — not a time slot. See the schema.
    course: str = "unknown"
    notes: str = ""

    @property
    def is_snack(self) -> bool:
        return self.course == "snack"

    def to_dict(self) -> dict:
        return {
            "items": [i.to_dict() for i in self.items],
            "course": self.course,
            "notes": self.notes,
        }


def sniff_mime(data: bytes) -> str:
    """Guess the image type from magic bytes.

    iOS Shortcuts will hand you HEIC as often as JPEG, and Gemini needs the
    mime type to be right.
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
            return "image/heic"
        if brand in (b"avif", b"avis"):
            return "image/avif"
    return "image/jpeg"


def active_model() -> str:
    """The model a call would actually use right now — the env override if
    one is set, otherwise the default. Public because the health check
    reports it: a pinned model that silently outranks DEFAULT_MODEL is
    exactly the kind of thing you want visible from outside the process."""
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def fallback_models() -> list[str]:
    """The chain tried after `active_model`, in order."""
    return [m for m in _fallback_models() if m != active_model()]


def _fallback_models() -> list[str]:
    raw = os.environ.get("GEMINI_FALLBACK_MODELS")
    if raw is not None:  # explicitly set — including "" to disable fallback
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_FALLBACK_MODELS)


def analyze_photo(
    image: bytes | str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    hint: Optional[str] = None,
    mime_type: Optional[str] = None,
    timeout: float = 12.0,
    budget: float = 30.0,
    http: Optional[httpx.Client] = None,
) -> PhotoAnalysis:
    """Identify foods in a photo.

    `image` is raw bytes or a base64 string (with or without a data: prefix).
    `hint` is optional free text from the user — "the chicken is 200g", "this
    is my post-workout shake" — which reliably beats guessing from pixels.

    On a capacity/rate-limit error, retries against each of
    GEMINI_FALLBACK_MODELS (or DEFAULT_FALLBACK_MODELS) in order before
    giving up. Any other kind of failure — a bad key, a bad image, a
    genuine bug in the request — fails immediately, since trying the same
    broken request against a different model wouldn't help.

    `timeout` caps one attempt; `budget` caps all of them together. Both
    matter, and the second one is the whole point: four models at a 60s
    timeout each is a four-minute worst case inside a function Vercel kills
    at 60s, so a slow — not failing — Gemini used to end as a 504 with a
    non-JSON body, after the diary writes had already happened. Falling
    back is only worth doing while there's time left to serve the answer.

    The per-attempt cap is tight on purpose. A refusal is not always fast:
    an overloaded endpoint took a measured 17.6s to answer 503, which is
    most of a budget spent learning nothing. The primary model answers in
    about two seconds, so anything still silent at twelve is not thinking
    hard — it's gone, and the next model is the better bet.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise VisionError("Missing GEMINI_API_KEY. Get one at https://aistudio.google.com/apikey")

    raw = _as_bytes(image)
    if not raw:
        raise VisionError("Empty image")

    primary = model or active_model()
    candidates = [primary] + [m for m in _fallback_models() if m != primary]

    client = http or httpx.Client(timeout=timeout)
    deadline = time.monotonic() + budget
    try:
        last_error: Optional[VisionError] = None
        for i, candidate_model in enumerate(candidates):
            left = deadline - time.monotonic()
            if left <= 1.0 and last_error is not None:
                log.warning("out of time after %s, not trying %s",
                            candidates[i - 1], candidate_model)
                break
            try:
                return _call_model(client, candidate_model, raw, key,
                                   hint=hint, mime_type=mime_type,
                                   timeout=min(timeout, max(left, 1.0)))
            except VisionOverloaded as e:
                last_error = e
                if i + 1 < len(candidates):
                    log.warning("%s overloaded, falling back to %s: %s",
                               candidate_model, candidates[i + 1], e)
                continue
        raise last_error or VisionError("No Gemini model configured")
    finally:
        if http is None:
            client.close()


def _call_model(
    client: httpx.Client,
    model: str,
    raw: bytes,
    key: str,
    *,
    hint: Optional[str],
    mime_type: Optional[str],
    timeout: Optional[float] = None,
) -> PhotoAnalysis:
    """One request against one model. Raises VisionOverloaded for a
    retryable failure, plain VisionError for anything else."""
    prompt = PROMPT
    if hint and hint.strip():
        # The user's own words are better evidence than the pixels. Say so
        # explicitly or the model averages them together.
        prompt += (
            f"\n\nThe person who took the photo says: \"{hint.strip()}\"\n"
            "Treat that as authoritative where it conflicts with what you see."
        )

    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": mime_type or sniff_mime(raw),
                    "data": base64.b64encode(raw).decode(),
                }},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
            "temperature": 0.2,
        },
    }

    # Thinking helps portion estimates but costs seconds, and a phone is
    # waiting on this.
    #
    # Sending nothing here does NOT mean thinking is off — that was the
    # assumption, and it was wrong. The Gemini 3.x thinking models default
    # to *dynamic* thinking, and they bill for it: gemini-3.7-flash reports
    # ~250 thinking tokens on a photo of a single apple, gemini-3.6-flash
    # ~620, and those tokens are most of why they take five to nine seconds
    # to say "apple". The Lite models this now defaults to report none at
    # all, which is the real reason they're fast.
    #
    # Nor can it simply be turned off: thinkingBudget=0 is rejected with a
    # 400 by gemini-3.6-flash and the Lite models alike. Picking a model
    # that doesn't think is the lever that actually exists; this knob only
    # goes the other way, for when you want the accuracy back.
    budget = os.environ.get("GEMINI_THINKING_BUDGET")
    if budget is not None:
        try:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(budget)}
        except ValueError:
            pass

    try:
        kwargs = {} if timeout is None else {"timeout": timeout}
        r = client.post(
            f"{API_ROOT}/{model}:generateContent",
            json=body,
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            **kwargs,
        )
    except httpx.HTTPError as e:
        # A dropped connection might well succeed against a different
        # endpoint/model, so this counts as retryable too.
        raise VisionOverloaded(f"{model}: could not reach Gemini: {e}") from e

    if r.status_code in (404, 429, 503):
        # 404 means "this model id doesn't exist (anymore)" — deprecated,
        # renamed, or restricted to existing users only, all of which
        # Google has actually done to models this app was pinned to. The
        # fix is the same as an overload: try the next one in the chain.
        raise VisionOverloaded(f"{model}: HTTP {r.status_code} (unavailable): {r.text[:200]}")
    if r.status_code >= 500:
        # Google's side breaking is exactly the "try elsewhere" case too.
        raise VisionOverloaded(f"{model}: HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        # 400/401/403 etc. — the request itself is the problem, and every
        # other model would reject it the same way.
        raise VisionError(f"{model}: HTTP {r.status_code}: {r.text[:300]}")

    return _parse_response(r.json())


def _parse_response(payload: dict) -> PhotoAnalysis:
    candidates = payload.get("candidates") or []
    if not candidates:
        # Usually a safety block or a bad prompt; the reason is worth showing.
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise VisionError(f"Gemini returned nothing ({reason})")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        finish = candidates[0].get("finishReason", "unknown")
        raise VisionError(f"Gemini returned an empty response (finishReason={finish})")

    try:
        data = json.loads(text)
    except ValueError as e:
        raise VisionError(f"Gemini did not return valid JSON: {text[:200]}") from e

    items = []
    for row in (data.get("items") or []):
        query = (row.get("query") or row.get("label") or "").strip()
        grams = _positive_float(row.get("grams"))
        if not query or grams is None:
            continue  # unusable rows are dropped, not guessed at
        items.append(FoodGuess(
            label=(row.get("label") or query).strip(),
            query=query,
            grams=grams,
            confidence=_clamp(row.get("confidence"), 0.0, 1.0, default=0.5),
            branded=bool(row.get("branded")),
            notes=(row.get("notes") or "").strip(),
            box_2d=_parse_box(row.get("box_2d")),
        ))

    # Anything that isn't one of the two answers we asked for is "unknown",
    # which the caller treats as "no opinion" — including a stale model
    # replying with an old-style time slot like "lunch".
    course = (data.get("course") or "unknown").strip().lower()
    if course not in ("meal", "snack"):
        course = "unknown"

    return PhotoAnalysis(
        items=items,
        course=course,
        notes=(data.get("notes") or "").strip(),
    )


def _as_bytes(image: bytes | str) -> bytes:
    if isinstance(image, bytes):
        return image
    text = str(image).strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        return base64.b64decode(text, validate=False)
    except Exception as e:
        raise VisionError(f"Could not decode image: {e}") from e


def _parse_box(v) -> Optional[tuple[int, int, int, int]]:
    """[ymin, xmin, ymax, xmax], each clamped to 0-1000. A malformed or
    missing box just means no marker for that item — never worth failing
    the whole photo over."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (max(0, min(1000, int(round(float(n))))) for n in v)
    except (TypeError, ValueError):
        return None
    if ymax <= ymin or xmax <= xmin:
        return None  # degenerate box — height or width is zero or negative
    return (ymin, xmin, ymax, xmax)


def _positive_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 1) if f > 0 else None


def _clamp(v, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default
