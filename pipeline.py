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
) -> dict:
    """Analyze a food photo and log what it finds.

    `meal` overrides both Gemini's guess and the time-of-day fallback.
    `dry_run` runs everything except the write, which is what you want the
    first few times you point a Shortcut at this. `always_log_uncertain`
    logs a flagged best-guess instead of holding it for review — see the
    module docstring for when to set it.
    """
    day = normalize_date(date)
    analysis = vision.analyze_photo(image, hint=hint)

    if not analysis.items:
        return _empty_result(day, meal, analysis, "No food found in that photo.")

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
    chosen_meal = explicit_meal or "auto"
    group_id = resolve_diary_group(chosen_meal)
    meal_label = group_name(group_id)

    own_client = client is None
    client = client or CronometerClient()
    try:
        # One login before fanning out, so parallel searches share a session
        # instead of racing to create four of them.
        client.ensure_auth()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            matches = list(pool.map(
                lambda g: _match_one(client, g),
                analysis.items,
            ))

        logged, review, failed = [], [], []
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

            if dry_run:
                entry["would_log"] = {
                    "food_id": hit.food_id, "food_name": hit.name,
                    "measure_id": hit.measure_id, "grams": guess.grams,
                    "date": day, "meal": meal_label,
                }
                logged.append(entry)
                continue

            try:
                written = client.add_entry(
                    food_id=hit.food_id,
                    measure_id=hit.measure_id,
                    grams=guess.grams,
                    date=day,
                    diary_group=meal_label,
                )
                entry["entry"] = {**written, "food_name": written.get("food_name") or hit.name}
                entry.pop("raw", None)
                logged.append(entry)
            except CronometerError as e:
                failed.append({**entry, "error": str(e)})

        daily = None
        if not dry_run and logged:
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
        "vision": {"meal_guess": analysis.meal, "notes": analysis.notes},
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
