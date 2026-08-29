"""
Plain-Python client for Cronometer's mobile API.

No MCP, no framework, no subscription needed — mobile.cronometer.com is the
same endpoint the free Android app talks to, so a normal email/password
account is enough.

Two protocols are in play, which is confusing until you've seen it:
  * /api/v2/<method>  — JSON-RPC-ish. Everything is a POST, and the session
                        lives in an "auth" block *inside the body*.
  * /api/v3/...       — actual REST. Session goes in the x-crono-session header.
Deletes are v3; everything else here is v2.

    client = CronometerClient()
    hits = client.search_foods("greek yogurt")
    client.add_entry(hits[0].food_id, hits[0].measure_id, grams=170)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

log = logging.getLogger("cronometer")

BASE_URL = "https://mobile.cronometer.com"

# The API cares that these look like a real app build. They're not secret and
# they don't need to match your actual phone — they just need to be present.
_APP_AUTH = {"api": 3, "os": "Android", "build": "2807", "flavour": "free"}
_APP_BUILD = "4.48.2 b2807-a"
_DEVICE = "Android 14 (SDK 34), Google Pixel 6 Pro"

# Cronometer's diary sections. 0 is the "uncategorized" bucket at the top of
# the day, which is where the app puts things you log without picking a meal.
DIARY_GROUPS = {
    "uncategorized": 0,
    "breakfast": 1,
    "lunch": 2,
    "dinner": 3,
    "snacks": 4,
}


class CronometerError(Exception):
    """Anything the API refused to do."""


class CronometerAuthError(CronometerError):
    """Bad credentials, or a session we couldn't renew."""


class CronometerRateLimited(CronometerError):
    """Cronometer is throttling us. Back off."""


# ──────────────────────────────────────────────────────────────────────
# time
# ──────────────────────────────────────────────────────────────────────

def local_timezone() -> ZoneInfo:
    """The user's timezone, or UTC if we genuinely can't tell.

    This matters more than it looks. Serverless containers run in UTC, so
    without an explicit CRONOMETER_TIMEZONE a 9pm dinner photo gets filed
    under tomorrow's date. Set the env var.
    """
    name = os.environ.get("CRONOMETER_TIMEZONE")
    if name:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            log.warning("CRONOMETER_TIMEZONE=%r is not a known IANA zone", name)

    # Fall back to whatever /etc/localtime points at (works on macOS + most Linux).
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return ZoneInfo(link.split("zoneinfo/", 1)[1])
    except (OSError, ZoneInfoNotFoundError):
        pass

    return ZoneInfo("UTC")


def today() -> str:
    """Today's date in the user's timezone, as YYYY-MM-DD."""
    return datetime.now(local_timezone()).date().isoformat()


def normalize_date(value: Optional[str]) -> str:
    """Accept YYYY-MM-DD, 'today', 'yesterday', or None."""
    if value is None or not str(value).strip():
        return today()
    v = str(value).strip().lower()
    if v == "today":
        return today()
    if v == "yesterday":
        d = datetime.now(local_timezone()).date() - timedelta(days=1)
        return d.isoformat()
    try:
        return datetime.strptime(v, "%Y-%m-%d").date().isoformat()
    except ValueError as e:
        raise ValueError(f"Bad date {value!r}; want YYYY-MM-DD, 'today' or 'yesterday'") from e


def resolve_diary_group(group: Optional[str], when: Optional[datetime] = None) -> int:
    """Map a meal name to Cronometer's group id.

    'auto' picks by clock time, because a photo taken at 7pm is almost
    certainly dinner and making the caller specify that defeats the point.
    Pass 'uncategorized' if you'd rather Cronometer not guess.
    """
    if group is None:
        group = "auto"
    key = str(group).strip().lower()
    if key in DIARY_GROUPS:
        return DIARY_GROUPS[key]
    if key in ("auto", ""):
        now = when or datetime.now(local_timezone())
        hour = now.hour + now.minute / 60
        if hour < 10.5:
            return DIARY_GROUPS["breakfast"]
        if hour < 15:
            return DIARY_GROUPS["lunch"]
        if hour < 21:
            return DIARY_GROUPS["dinner"]
        return DIARY_GROUPS["snacks"]
    raise ValueError(f"Unknown diary_group {group!r}; want auto/{'/'.join(DIARY_GROUPS)}")


def group_name(group_id: int) -> str:
    for name, gid in DIARY_GROUPS.items():
        if gid == group_id:
            return name
    return f"group_{group_id}"


# ──────────────────────────────────────────────────────────────────────
# result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FoodHit:
    """One row from a food search."""
    food_id: int
    name: str
    source: str
    measure_id: Optional[int]
    measure_display: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiaryEntry:
    entry_id: Optional[int]      # aka servingId; needed to delete
    food_id: Optional[int]
    food_name: str
    measure_id: Optional[int]
    grams: float
    diary_group: int

    @property
    def meal(self) -> str:
        return group_name(self.diary_group)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["meal"] = self.meal
        return d


# ──────────────────────────────────────────────────────────────────────
# session cache
# ──────────────────────────────────────────────────────────────────────

class _SessionCache:
    """Persists the session key so we don't re-login on every cold start.

    Cronometer locks you out with "Too Many Attempts" if you hammer /login,
    which a serverless function will absolutely do without this. Warm
    invocations share /tmp, so the cache survives between requests.
    """

    def __init__(self, email: str, ttl_seconds: int = 12 * 3600, directory: Optional[str] = None):
        self.ttl = ttl_seconds
        digest = hashlib.sha256(email.encode()).hexdigest()[:16]
        base = Path(directory or os.environ.get("CRONOMETER_CACHE_DIR") or tempfile.gettempdir())
        self.path = base / f"cronometer-session-{digest}.json"

    def load(self) -> Optional[tuple[int, str]]:
        try:
            blob = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - blob.get("created_at", 0) > self.ttl:
            return None
        if not blob.get("user_id") or not blob.get("token"):
            return None
        return int(blob["user_id"]), str(blob["token"])

    def save(self, user_id: int, token: str) -> None:
        try:
            self.path.write_text(json.dumps(
                {"user_id": user_id, "token": token, "created_at": time.time()}
            ))
            os.chmod(self.path, 0o600)
        except OSError as e:  # read-only fs is survivable, just slower
            log.debug("could not cache session: %s", e)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────
# client
# ──────────────────────────────────────────────────────────────────────

class CronometerClient:
    """Talks to Cronometer. Logs in lazily on the first call that needs auth."""

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        *,
        timezone: Optional[str] = None,
        timeout: float = 30.0,
        use_session_cache: bool = True,
        http: Optional[httpx.Client] = None,
    ):
        self.email = email or os.environ.get("CRONOMETER_EMAIL") or os.environ.get("CRONOMETER_USERNAME")
        self.password = password or os.environ.get("CRONOMETER_PASSWORD")
        if not self.email or not self.password:
            raise CronometerAuthError(
                "Missing credentials. Set CRONOMETER_EMAIL and CRONOMETER_PASSWORD "
                "(in .env for local runs, or project env vars when deployed)."
            )
        self.timezone = timezone or os.environ.get("CRONOMETER_TIMEZONE") or str(local_timezone())
        self._http = http or httpx.Client(timeout=timeout)
        self._owns_http = http is None
        self._cache = _SessionCache(self.email) if use_session_cache else None

        self.user_id: Optional[int] = None
        self._token: Optional[str] = None

    # -- lifecycle ----------------------------------------------------

    def __enter__(self) -> "CronometerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    # -- auth ---------------------------------------------------------

    @property
    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self._token:
            h["x-crono-session"] = self._token
        return h

    def _auth_block(self) -> dict:
        return {"userId": self.user_id, "token": self._token, **_APP_AUTH}

    def ensure_auth(self) -> None:
        if self._token:
            return
        if self._cache:
            cached = self._cache.load()
            if cached:
                self.user_id, self._token = cached
                log.debug("reusing cached session for user %s", self.user_id)
                return
        self.login()

    def login(self) -> int:
        """Exchange email/password for a session key. Returns the user id."""
        payload = {
            "email": self.email,
            "password": self.password,
            "timezone": self.timezone,
            "userCode": None,
            "build": _APP_BUILD,
            "device": _DEVICE,
            "firebaseToken": "",
            "features": {},
            "auth": {"userId": None, "token": None, **_APP_AUTH},
            "lastSeen": 0,
            "config": {"call_version": 2},
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                r = self._http.post(f"{BASE_URL}/api/v2/login", json=payload)
                r.raise_for_status()
                data = r.json()

                # The API returns HTTP 200 with result=FAIL for bad logins.
                if isinstance(data, dict) and data.get("result") == "FAIL":
                    err = str(data.get("error", "unknown error"))
                    if "Too Many Attempts" in err:
                        raise CronometerRateLimited(err)
                    raise CronometerAuthError(f"Login rejected: {err}")

                self.user_id = int(data["id"])
                self._token = str(data["sessionKey"])
                if self._cache:
                    self._cache.save(self.user_id, self._token)
                log.info("logged in as user %s", self.user_id)
                return self.user_id

            except CronometerAuthError:
                raise  # wrong password won't fix itself
            except (CronometerRateLimited, httpx.HTTPError, KeyError) as e:
                last_error = e
                if attempt < 2:
                    _sleep_backoff(attempt)
                    continue

        raise CronometerError(f"Login failed after 3 attempts: {last_error}")

    # -- transport ----------------------------------------------------

    def _v2(self, method: str, **body: Any) -> Any:
        """POST to a v2 method, retrying once through a fresh login on 401/403."""
        self.ensure_auth()

        for attempt in range(3):
            payload = {"auth": self._auth_block(), **body}
            try:
                r = self._http.post(f"{BASE_URL}/api/v2/{method}", json=payload, headers=self._headers)
            except httpx.HTTPError as e:
                if attempt < 2:
                    _sleep_backoff(attempt)
                    continue
                raise CronometerError(f"{method}: network error: {e}") from e

            if r.status_code in (401, 403):
                log.info("%s: session expired, re-authenticating", method)
                self._token = None
                if self._cache:
                    self._cache.clear()
                self.login()
                continue

            if r.status_code == 429:
                if attempt < 2:
                    _sleep_backoff(attempt)
                    continue
                raise CronometerRateLimited(f"{method}: rate limited")

            if r.status_code >= 500 and attempt < 2:
                _sleep_backoff(attempt)
                continue

            if r.status_code != 200:
                raise CronometerError(f"{method}: HTTP {r.status_code}: {r.text[:200]}")

            try:
                data = r.json()
            except ValueError:
                return {}

            if isinstance(data, dict) and data.get("result") == "FAIL":
                raise CronometerError(f"{method}: {data.get('error', 'unknown error')}")
            return data

        raise CronometerError(f"{method}: gave up after 3 attempts")

    # -- reads --------------------------------------------------------

    def search_foods(self, query: str, limit: int = 15) -> list[FoodHit]:
        """Search the food database.

        `score` is Cronometer's own relevance number — it is not normalized
        and only means anything relative to other hits for the same query.
        """
        if not query or not query.strip():
            return []
        data = self._v2("find_food", query=query.strip(), tab="ALL", sources=["All"])
        hits = []
        for row in (data.get("foods") or [])[:limit]:
            try:
                food_id = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            hits.append(FoodHit(
                food_id=food_id,
                name=row.get("name") or "",
                source=row.get("source") or "",
                measure_id=_maybe_int(row.get("measureId")),
                measure_display=row.get("measureDisplayName") or "",
                score=float(row.get("score") or 0.0),
            ))
        return hits

    def get_food(self, food_id: int) -> dict:
        """Full food record: every measure and every nutrient."""
        f = self._v2("get_food", id=int(food_id))
        measures = [
            {"measure_id": _maybe_int(m.get("id")),
             "name": m.get("displayName") or "",
             "grams": m.get("amount") or 0}
            for m in (f.get("measures") or [])
        ]
        nutrients = {}
        for n in (f.get("nutrients") or []):
            key = n.get("name") or f"nutrient_{n.get('nutrientId')}"
            nutrients[key] = {"amount": n.get("amount", 0), "unit": n.get("unit", "")}
        return {
            "food_id": int(food_id),
            "name": f.get("name") or "",
            "source": f.get("source") or "",
            "measures": measures,
            "nutrients": nutrients,
        }

    def find_gram_measure(self, food_id: int) -> Optional[int]:
        """The measure whose unit is plain grams, if the food has one.

        Search results hand back a *default* measure (often "1 cup"), and
        add_entry's `grams` is interpreted against that measure. Passing an
        explicit gram measure is the unambiguous option when you're logging
        an estimated weight rather than a countable serving.
        """
        for m in self.get_food(food_id)["measures"]:
            name = (m["name"] or "").strip().lower()
            if name in ("g", "gram", "grams", "gramme", "grammes"):
                return m["measure_id"]
        return None

    def get_diary(self, date: Optional[str] = None) -> list[DiaryEntry]:
        """Food entries logged on a date."""
        day = normalize_date(date)
        data = self._v2("get_diary", day=day)
        entries = []
        for e in (data.get("diary") or []):
            if e.get("type") != "Serving":
                continue  # notes, biometrics, exercise all share this list
            entries.append(DiaryEntry(
                entry_id=_maybe_int(e.get("servingId")),
                food_id=_maybe_int(e.get("foodId")),
                food_name=e.get("foodName") or "",
                measure_id=_maybe_int(e.get("measureId")),
                grams=float(e.get("grams") or 0),
                # `order` packs the group into the high bits; `diaryGroup` is
                # not always present on read, so derive it when it's missing.
                diary_group=_maybe_int(e.get("diaryGroup"))
                if e.get("diaryGroup") is not None
                else (_maybe_int(e.get("order")) or 0) >> 16,
            ))
        return entries

    def get_daily_nutrition(self, date: Optional[str] = None) -> dict:
        """Consumed vs. target macros for a date, plus the entry list.

        get_diary carries what you actually ate; get_nutrients carries the
        RDI targets. Neither is useful alone, so this returns both.
        """
        day = normalize_date(date)
        raw = self._v2("get_diary", day=day)
        summary = raw.get("summary") or {}
        consumed = summary.get("consumed") or {}
        macros = summary.get("macros") or {}

        try:
            targets = self._v2("get_nutrients", day=day)
        except CronometerError as e:
            log.warning("get_nutrients failed for %s: %s", day, e)
            targets = {}

        entries = [
            DiaryEntry(
                entry_id=_maybe_int(e.get("servingId")),
                food_id=_maybe_int(e.get("foodId")),
                food_name=e.get("foodName") or "",
                measure_id=_maybe_int(e.get("measureId")),
                grams=float(e.get("grams") or 0),
                diary_group=_maybe_int(e.get("diaryGroup"))
                if e.get("diaryGroup") is not None
                else (_maybe_int(e.get("order")) or 0) >> 16,
            )
            for e in (raw.get("diary") or [])
            if e.get("type") == "Serving"
        ]

        eaten = {
            "calories": _num(consumed.get("total")),
            "protein_g": _num(consumed.get("protein_g")),
            "carbs_g": _num(consumed.get("carbs_g")),
            "fat_g": _num(consumed.get("fat_g")),
        }
        target = {
            "calories": _num(macros.get("energy")),
            "protein_g": _num(macros.get("protein")),
            "carbs_g": _num(macros.get("carbs")),
            "fat_g": _num(macros.get("fat")),
        }
        remaining = {
            k: round(target[k] - eaten[k], 1) if target[k] else None
            for k in eaten
        }

        return {
            "date": day,
            "consumed": eaten,
            "targets": target,
            "remaining": remaining,
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in entries],
            "nutrient_targets": targets,
        }

    # -- writes -------------------------------------------------------

    def add_entry(
        self,
        food_id: int,
        measure_id: int,
        grams: float,
        date: Optional[str] = None,
        diary_group: Optional[str] = "auto",
        *,
        skip_if_duplicate: bool = True,
    ) -> dict:
        """Log a serving. Returns {"entry_id": int|None, ...}.

        `skip_if_duplicate` checks the live diary rather than an in-process
        cache, so it still works when every request is a fresh process (as
        on Vercel). Double-logging your lunch because a Shortcut retried is
        an easy and annoying failure.
        """
        day = normalize_date(date)
        group = resolve_diary_group(diary_group)
        grams = round(float(grams), 2)
        if grams <= 0:
            raise ValueError("grams must be positive")

        if skip_if_duplicate:
            for e in self.get_diary(day):
                if (e.food_id == int(food_id)
                        and e.measure_id == int(measure_id)
                        and abs(e.grams - grams) < 0.5
                        and e.diary_group == group):
                    return {
                        "entry_id": e.entry_id,
                        "skipped": True,
                        "reason": "already_logged",
                        "date": day,
                        "meal": group_name(group),
                        "food_name": e.food_name,
                        "grams": e.grams,
                    }

        serving = {
            "userId": self.user_id,
            "foodId": int(food_id),
            "measureId": int(measure_id),
            "grams": grams,
            "day": day,
            "diaryGroup": group,
            # Low 16 bits are the position inside the group; the server
            # re-sorts, so 1 is fine for an append.
            "order": (group << 16) | 1,
            "idempotencyKey": str(uuid.uuid4()),
        }
        result = self._v2("add_serving", serving=serving)

        return {
            "entry_id": _extract_entry_id(result),
            "skipped": False,
            "date": day,
            "meal": group_name(group),
            "food_id": int(food_id),
            "measure_id": int(measure_id),
            "grams": grams,
            "raw": result,
        }

    def remove_entries(self, entry_ids: list[int], date: Optional[str] = None) -> dict:
        """Delete diary entries by id.

        The v3 delete endpoint wants the whole entry objects back, not just
        ids, so we re-read the day and echo the matching ones.
        """
        day = normalize_date(date)
        wanted = {str(i) for i in entry_ids}
        raw = self._v2("get_diary", day=day)
        to_delete = [e for e in (raw.get("diary") or []) if str(e.get("servingId")) in wanted]
        if not to_delete:
            return {"removed": [], "count": 0, "note": f"No matching entries on {day}"}

        self.ensure_auth()
        r = self._http.request(
            "DELETE",
            f"{BASE_URL}/api/v3/user/{self.user_id}/diary-entries",
            json={"diaryEntries": to_delete},
            headers=self._headers,
        )
        if r.status_code not in (200, 204):
            raise CronometerError(f"delete failed: HTTP {r.status_code}: {r.text[:200]}")
        removed = [_maybe_int(e.get("servingId")) for e in to_delete]
        return {"removed": removed, "count": len(removed), "date": day}


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter, so parallel callers don't sync up."""
    time.sleep(min(2 ** attempt * 2.0, 10.0) + random.uniform(0, 0.5))


def _maybe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return default


def _extract_entry_id(result: Any) -> Optional[int]:
    """add_serving's shape varies; dig the id out of whatever came back."""
    if isinstance(result, (int, str)):
        return _maybe_int(result)
    if isinstance(result, dict):
        for key in ("servingId", "id", "entryId", "serving_id"):
            got = _maybe_int(result.get(key))
            if got is not None:
                return got
        for nested in ("serving", "entry", "result"):
            if isinstance(result.get(nested), dict):
                got = _extract_entry_id(result[nested])
                if got is not None:
                    return got
    return None
