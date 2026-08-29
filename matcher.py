"""
Picks the right Cronometer food for a guess like "grilled chicken breast".

Cronometer's own `score` is a decent relevance signal but it's tuned for a
human scrolling a list — it happily puts "Chicken Breast, Breaded, Frozen"
above the plain one. Vision guesses are fuzzy in a different way, so we blend
three signals:

  1. Cronometer's score, normalized against the best hit for this query.
  2. Token F1 between the guess and the food name. F1, not substring match,
     because it punishes *extra* words: "chicken breast" vs "chicken breast
     nuggets, breaded" should lose points for the words we didn't ask for.
  3. Source preference. For a photo of food on a plate, a generic USDA/NCCDB
     entry beats a branded one. If the vision step saw a package, that flips.

If the winner is weak, or barely ahead of second place, we say so rather than
logging it — `needs_review` is the signal for "ask the human first".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from cronometer_client import FoodHit

# Words that carry no discriminating power in food names.
_STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "with", "without", "in", "on",
    "from", "to", "for", "plus", "type", "style", "all", "any",
}

# Cronometer source names, roughly. Generic reference databases first: they
# describe "chicken breast" rather than "Brand X Chicken Breast Product".
_GENERIC_SOURCES = {"usda", "nccdb", "srlegacy", "sr legacy", "foundation", "cronometer"}
_BRANDED_SOURCES = {"crdb", "branded", "restaurant", "openfoodfacts", "off", "verified"}

# Weights. They sum to 1.0; tune them here rather than sprinkling magic
# numbers through the scoring function.
W_CRONO = 0.40
W_NAME = 0.45
W_SOURCE = 0.15

# Below this, we don't trust the pick. Above it but within MIN_MARGIN of the
# runner-up means it's a coin flip between two plausible foods.
MIN_CONFIDENCE = 0.45
MIN_MARGIN = 0.06


@dataclass
class ScoredHit:
    hit: FoodHit
    confidence: float
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.hit.to_dict()
        d["confidence"] = round(self.confidence, 3)
        d["breakdown"] = {k: round(v, 3) for k, v in self.breakdown.items()}
        return d


@dataclass
class MatchResult:
    query: str
    best: Optional[ScoredHit]
    alternatives: list[ScoredHit]
    needs_review: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "best": self.best.to_dict() if self.best else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "needs_review": self.needs_review,
            "reason": self.reason,
        }


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens, minus stopwords and pure punctuation."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def token_f1(query: str, candidate: str) -> float:
    """Harmonic mean of precision and recall over word tokens.

    Recall: how much of what we asked for is present.
    Precision: how much of the food name we actually asked for.
    A name with lots of unrequested qualifiers scores low on precision, which
    is exactly the "breaded, frozen, with sauce" case we want to demote.

    Known gap: this is spelling-exact, so "yoghurt" and "yogurt" share
    nothing. Cronometer's own score handles fuzzy spelling, which is a good
    part of why it keeps a 40% weight rather than being a tiebreaker.
    """
    q, c = tokenize(query), tokenize(candidate)
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    if not overlap:
        return 0.0
    precision = overlap / len(c)
    recall = overlap / len(q)
    return 2 * precision * recall / (precision + recall)


def source_weight(source: str, prefer_branded: bool = False) -> float:
    """How much we trust this database for this kind of guess."""
    s = (source or "").strip().lower()
    generic = any(s.startswith(g) or g in s for g in _GENERIC_SOURCES)
    branded = any(s.startswith(b) or b in s for b in _BRANDED_SOURCES)

    if prefer_branded:
        if branded:
            return 1.0
        return 0.6 if generic else 0.5
    if generic:
        return 1.0
    if branded:
        return 0.55
    return 0.7  # unknown source; neither rewarded nor punished


def score_hits(
    query: str,
    hits: Sequence[FoodHit],
    *,
    prefer_branded: bool = False,
) -> list[ScoredHit]:
    """Rank search results against the query. Best first."""
    if not hits:
        return []

    # Cronometer's score has no fixed range, so normalize within this result
    # set. If every hit scored 0 we just drop that signal entirely rather
    # than dividing by zero or pretending everything is a perfect match.
    top = max((h.score for h in hits), default=0.0)
    crono_usable = top > 0

    scored = []
    for h in hits:
        crono = (h.score / top) if crono_usable else 0.0
        name = token_f1(query, h.name)
        src = source_weight(h.source, prefer_branded)

        if crono_usable:
            confidence = W_CRONO * crono + W_NAME * name + W_SOURCE * src
        else:
            # Redistribute Cronometer's share onto the two signals we have.
            confidence = (W_NAME * name + W_SOURCE * src) / (W_NAME + W_SOURCE)

        # A hit with no usable measure can't be logged, so it can't win.
        if h.measure_id is None:
            confidence *= 0.5

        scored.append(ScoredHit(
            hit=h,
            confidence=confidence,
            breakdown={"cronometer": crono, "name": name, "source": src},
        ))

    scored.sort(key=lambda s: s.confidence, reverse=True)
    return scored


def match(
    query: str,
    hits: Sequence[FoodHit],
    *,
    prefer_branded: bool = False,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    keep_alternatives: int = 3,
) -> MatchResult:
    """Rank hits and decide whether the winner is trustworthy."""
    scored = score_hits(query, hits, prefer_branded=prefer_branded)

    if not scored:
        return MatchResult(query, None, [], True, "no search results")

    best = scored[0]
    alts = scored[1:1 + keep_alternatives]

    if best.hit.measure_id is None:
        return MatchResult(query, best, alts, True, "best match has no serving measure")
    if best.confidence < min_confidence:
        return MatchResult(
            query, best, alts, True,
            f"low confidence ({best.confidence:.2f} < {min_confidence:.2f})",
        )
    if alts and (best.confidence - alts[0].confidence) < min_margin:
        return MatchResult(
            query, best, alts, True,
            f"ambiguous: '{best.hit.name}' vs '{alts[0].hit.name}'",
        )

    return MatchResult(query, best, alts, False, "")


def match_food(
    client,
    query: str,
    *,
    prefer_branded: bool = False,
    limit: int = 15,
    **kwargs,
) -> MatchResult:
    """Search + rank in one call. `client` is a CronometerClient."""
    return match(query, client.search_foods(query, limit=limit),
                 prefer_branded=prefer_branded, **kwargs)
