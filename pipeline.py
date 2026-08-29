"""
Photo in, diary entries out.

    result = log_photo(open("lunch.jpg", "rb").read(), meal="lunch")
    print(result["summary"])

Two ways to run this, picked with `always_log_uncertain`:

  - False (default; CLI, Shortcut): conservative. Anything the matcher
    flags as uncertain lands in `needs_review` instead of the diary.
    Correcting a wrong entry is more annoying than adding a missing one,
    so when in doubt this does nothing and reports why.

  - True (the camera page): every item that has *any* database match gets
    logged — flagged, not skipped, when uncertain — because that page has
    no confirm-tap to spend on doubt. The tradeoff moves from "don't log
    the wrong thing" to "never block the shutter," and undo/swap in the
    UI are what make that safe: fixing a flagged entry after the fact
    costs one tap, the same as confirming it would have.

The work splits in two, and callers can take it either whole or in halves:

    analyze()  photo → what's on the plate, resolved to database rows.
               Reads only; writes nothing.
    commit()   those rows → diary entries.

`log_photo` is both back to back, which is what the CLI and the Shortcut
want: one call, one answer. The camera page calls them as two requests
instead, so it can draw detection markers the moment `analyze` returns
rather than holding a frozen frame through the writes as well. Splitting
also makes a retry safe — `commit` re-sends the exact grams `analyze`
decided on, so the duplicate check actually matches, where re-running the
whole pipeline would roll a slightly different portion estimate and log
your lunch twice.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import matcher
import vision
from cronometer_client import CronometerClient, CronometerError, normalize_date, resolve_diary_group, group_name

log = logging.getLogger("crono_vision")

# Vision guesses below this confidence never get logged automatically, even
# if the food match itself is clean — a confidently-matched wrong food is
# the worst outcome.
MIN_VISION_CONFIDENCE = 0.35


def analyze(
    image: bytes | str,
    *,
    date: Optional[str] = None,
    meal: Optional[str] = None,
    hint: Optional[str] = None,
    always_log_uncertain: bool = False,
    client: Optional[CronometerClient] = None,
    min_vision_confidence: float = MIN_VISION_CONFIDENCE,
    max_workers: int = 4,
) -> dict:
    """Work out what's in the photo and which database row each item is.

    Writes nothing. Returns the same `needs_review` / `failed` buckets
    `log_photo` does, plus `pending`: the items that would be logged, each
    carrying a `plan` of exactly what to write. Hand that straight to
    `commit`.

    A photo with no food in it returns the finished empty result — summary
    and all — since there's nothing left for a second phase to do.
    """
    day = normalize_date(date)

    # Meal precedence: caller's explicit choice, else the clock. "auto" is
    # a sentinel for "no real preference", same as not passing meal at all.
    #
    # This deliberately ignores what Gemini thinks the food looks like
    # (analysis.meal) even though it's right there — that was tried, and
    # produced exactly the surprise it sounds like: a steak dinner
    # photographed at 12:27am logged as "lunch" because the plate read as
    # a lunch-type meal, while a can of soda at the same hour logged as a
    # snack. Every other diet tracker buckets by time slot, not food
    # content, and that's the less surprising default — "late dinner"
    # beats "lunch at 1am" even when the plate genuinely looks like lunch.
    explicit_meal = meal if meal and meal.strip().lower() != "auto" else None
    group_id = resolve_diary_group(explicit_meal or "auto")
    meal_label = group_name(group_id)

    own_client = client is None
    client = client or CronometerClient()
    try:
        # Log in while Gemini is still looking at the photo. These two have
        # nothing to say to each other, and on a cold instance the login is
        # a full round trip to mobile.cronometer.com that used to wait its
        # turn behind a call already taking several seconds. Overlapping
        # them makes it free.
        with ThreadPoolExecutor(max_workers=max(2, max_workers)) as pool:
            auth = pool.submit(client.ensure_auth)
            # If this raises, the `with` still joins the login on the way
            # out — the client never gets closed from under a live request.
            analysis = vision.analyze_photo(image, hint=hint)

            if not analysis.items:
                auth.result()
                return {**_empty_result(day, meal, analysis,
                                        "No food found in that photo."),
                        "pending": []}

            # Searches ride the session this created, so it has to be done
            # — and this is where a bad password surfaces.
            auth.result()

            matches = list(pool.map(
                lambda g: _match_one(client, g),
                analysis.items,
            ))
    finally:
        if own_client:
            client.close()

    pending, review, failed = [], [], []
    for guess, result in zip(analysis.items, matches):
        entry = {"guess": guess.to_dict()}

        if isinstance(result, Exception):
            failed.append({**entry, "error": str(result)})
            continue

        entry["match"] = result.to_dict()

        # Vision confidence is checked before the database result on
        # purpose. If we don't trust what the food *is*, "nothing
        # matched 'unidentifiable sauce'" blames the wrong step — the
        # honest answer is that we couldn't tell what we were looking at.
        uncertain_reason = None
        if guess.confidence < min_vision_confidence:
            uncertain_reason = f"unsure this is {guess.label} ({guess.confidence:.0%})"
        elif result.best is None:
            failed.append({**entry, "error": f"nothing in the database matched '{guess.query}'"})
            continue
        elif result.needs_review:
            uncertain_reason = result.reason

        if uncertain_reason and not (always_log_uncertain and result.best is not None):
            entry["reason"] = uncertain_reason
            review.append(entry)
            continue

        # Either a clean match, or an uncertain one we're logging anyway
        # with a flag attached — same write path either way.
        hit = result.best.hit
        if uncertain_reason:
            entry["reason"] = uncertain_reason
            entry["flagged"] = True

        entry["plan"] = {
            "food_id": hit.food_id,
            "food_name": hit.name,
            "measure_id": hit.measure_id,
            "grams": guess.grams,
        }
        pending.append(entry)

    return {
        "date": day,
        "meal": meal_label,
        "vision": {"meal_guess": analysis.meal, "notes": analysis.notes},
        "pending": pending,
        "needs_review": review,
        "failed": failed,
    }


def commit(
    plans: list[dict],
    *,
    date: Optional[str] = None,
    meal: Optional[str] = None,
    client: Optional[CronometerClient] = None,
    max_workers: int = 4,
    dedupe: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Write `analyze`'s `pending` entries to the diary.

    Returns (logged, failed), both in the order they came in — one bad food
    doesn't sink the rest.

    Two things make this fast. The duplicate check reads the day once and
    shares that snapshot across every write, instead of re-reading an
    identical diary per item. And the writes themselves go out together,
    since they don't depend on each other. A four-item plate used to be
    eight serial round trips — read, write, read, write… — and is now two
    waves: one read, then four writes at once.

    Sharing one snapshot does mean two genuinely identical items in the
    same photo both get written, where the old per-item re-read would have
    collapsed the second into the first. That's the better answer anyway —
    two eggs on a plate are two eggs — and the case the check exists for, a
    retried request, is unaffected because the snapshot is read fresh each
    time.
    """
    if not plans:
        return [], []

    day = normalize_date(date)
    meal_label = group_name(resolve_diary_group(meal or "auto"))

    own_client = client is None
    client = client or CronometerClient()
    try:
        client.ensure_auth()
        snapshot = client.get_diary(day) if dedupe else None

        def write(entry: dict):
            p = entry["plan"]
            try:
                return client.add_entry(
                    food_id=p["food_id"],
                    measure_id=p["measure_id"],
                    grams=p["grams"],
                    date=day,
                    diary_group=meal_label,
                    skip_if_duplicate=dedupe,
                    diary=snapshot,
                )
            except CronometerError as e:
                return e

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(plans)))) as pool:
            results = list(pool.map(write, plans))
    finally:
        if own_client:
            client.close()

    logged, failed = [], []
    for entry, result in zip(plans, results):
        base = {k: v for k, v in entry.items() if k != "plan"}
        if isinstance(result, Exception):
            failed.append({**base, "error": str(result)})
            continue
        base["entry"] = {**result,
                         "food_name": result.get("food_name") or entry["plan"]["food_name"]}
        base.pop("raw", None)
        logged.append(base)
    return logged, failed


def log_photo(
    image: bytes | str,
    *,
    date: Optional[str] = None,
    meal: Optional[str] = None,
    hint: Optional[str] = None,
    dry_run: bool = False,
    always_log_uncertain: bool = False,
    client: Optional[CronometerClient] = None,
    min_vision_confidence: float = MIN_VISION_CONFIDENCE,
    max_workers: int = 4,
    with_daily: bool = True,
) -> dict:
    """Analyze a food photo and log what it finds.

    `meal` overrides both Gemini's guess and the time-of-day fallback.
    `dry_run` runs everything except the write, which is what you want the
    first few times you point a Shortcut at this. `always_log_uncertain`
    logs a flagged best-guess instead of holding it for review — see the
    module docstring for when to set it.

    `with_daily` fetches the day's running totals for the summary line. The
    Shortcut's notification uses them; anything with its own totals display
    should turn this off and fetch them separately, since it's two more
    round trips between the writes landing and the caller hearing about it.
    """
    own_client = client is None
    client = client or CronometerClient()
    try:
        found = analyze(
            image,
            date=date, meal=meal, hint=hint,
            always_log_uncertain=always_log_uncertain,
            client=client,
            min_vision_confidence=min_vision_confidence,
            max_workers=max_workers,
        )
        # An empty photo comes back already finished, summary and all.
        if "summary" in found:
            found.pop("pending", None)
            return found

        pending = found["pending"]
        review, failed = found["needs_review"], list(found["failed"])
        day, meal_label = found["date"], found["meal"]

        if dry_run:
            logged = []
            for entry in pending:
                plan = entry.pop("plan")
                entry["would_log"] = {
                    "food_id": plan["food_id"], "food_name": plan["food_name"],
                    "measure_id": plan["measure_id"], "grams": plan["grams"],
                    "date": day, "meal": meal_label,
                }
                logged.append(entry)
        else:
            logged, write_failures = commit(
                pending, date=day, meal=meal_label,
                client=client, max_workers=max_workers,
            )
            failed.extend(write_failures)

        daily = None
        if with_daily and not dry_run and logged:
            try:
                daily = client.get_daily_nutrition(day)
                daily.pop("nutrient_targets", None)  # too big for a phone response
                daily.pop("entries", None)
            except CronometerError as e:
                log.warning("could not read daily totals: %s", e)
    finally:
        if own_client:
            client.close()

    return {
        "date": day,
        "meal": meal_label,
        "dry_run": dry_run,
        "vision": found["vision"],
        "logged": logged,
        "needs_review": review,
        "failed": failed,
        "daily": daily,
        "summary": _summarize(logged, review, failed, meal_label, daily, dry_run),
    }


def _match_one(client: CronometerClient, guess: vision.FoodGuess):
    """Search + rank one guess. Exceptions come back as values, not raises,
    so one bad food doesn't sink the whole photo."""
    try:
        return matcher.match_food(client, guess.query, prefer_branded=guess.branded)
    except Exception as e:  # noqa: BLE001 - surfaced per-item in the result
        log.warning("match failed for %r: %s", guess.query, e)
        return e


def _empty_result(day: str, meal: Optional[str], analysis, summary: str) -> dict:
    return {
        "date": day,
        "meal": group_name(resolve_diary_group(meal or "auto")),
        "dry_run": False,
        "vision": {"meal_guess": analysis.meal, "notes": analysis.notes},
        "logged": [], "needs_review": [], "failed": [],
        "daily": None,
        "summary": summary,
    }


def _summarize(logged, review, failed, meal, daily, dry_run) -> str:
    """A sentence or two, short enough for a phone notification."""
    parts = []

    if logged:
        names = []
        flagged_count = 0
        for e in logged:
            src = e.get("would_log") or e.get("entry") or {}
            name = src.get("food_name") or e["guess"]["label"]
            mark = "?" if e.get("flagged") else ""
            names.append(f"{name}{mark} ({src.get('grams', e['guess']['grams']):.0f}g)")
            flagged_count += 1 if e.get("flagged") else 0
        verb = "Would log" if dry_run else "Logged"
        parts.append(f"{verb} to {meal}: " + ", ".join(names) + ".")
        if flagged_count:
            parts.append(f"({flagged_count} unsure — check with a tap.)")
    else:
        parts.append("Nothing logged.")

    if review:
        labels = ", ".join(e["guess"]["label"] for e in review)
        parts.append(f"Needs a check: {labels}.")
    if failed:
        labels = ", ".join(e["guess"]["label"] for e in failed)
        parts.append(f"Couldn't log: {labels}.")

    if daily and daily.get("consumed"):
        c = daily["consumed"]
        line = f"Today: {c['calories']:.0f} kcal, {c['protein_g']:.0f}g protein"
        left = (daily.get("remaining") or {}).get("calories")
        if left is not None:
            line += f" ({left:.0f} kcal left)"
        parts.append(line + ".")

    return " ".join(parts)
