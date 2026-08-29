#!/usr/bin/env python3
"""
Local CLI — the fastest way to check that everything works before deploying.

    python cli.py search "greek yogurt"
    python cli.py photo lunch.jpg --dry-run
    python cli.py photo lunch.jpg --meal dinner --hint "the rice is about 200g"
    python cli.py today
    python cli.py add 12345 6789 150 --meal lunch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env sits next to this file and is gitignored

import matcher  # noqa: E402
from cronometer_client import CronometerClient, CronometerError  # noqa: E402
from pipeline import log_photo  # noqa: E402
from vision import VisionError  # noqa: E402


def cmd_search(args) -> int:
    with CronometerClient() as client:
        hits = client.search_foods(args.query, limit=args.limit)
        if not hits:
            print("No results.")
            return 1
        result = matcher.match(args.query, hits, prefer_branded=args.branded)
        print(f"{'conf':>6}  {'score':>6}  {'food_id':>9}  {'meas':>6}  source      name")
        for s in matcher.score_hits(args.query, hits, prefer_branded=args.branded):
            h = s.hit
            print(f"{s.confidence:6.3f}  {h.score:6.1f}  {h.food_id:9d}  "
                  f"{str(h.measure_id or '-'):>6}  {h.source[:10]:<10}  {h.name}")
        print()
        if result.needs_review:
            print(f"⚠  would not auto-log: {result.reason}")
        else:
            print(f"✓  picks: {result.best.hit.name}  ({result.best.hit.measure_display})")
    return 0


def cmd_photo(args) -> int:
    data = Path(args.path).read_bytes()
    try:
        result = log_photo(data, date=args.date, meal=args.meal,
                           hint=args.hint, dry_run=args.dry_run)
    except VisionError as e:
        print(f"vision failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(result["summary"])
    for e in result["needs_review"]:
        best = (e.get("match") or {}).get("best")
        guess = e["guess"]
        print(f"\n?  {guess['label']} ({guess['grams']:.0f}g) — {e['reason']}")
        if best:
            print(f"   best guess: {best['name']}  [food_id {best['food_id']} "
                  f"measure {best['measure_id']}]")
        for alt in ((e.get("match") or {}).get("alternatives") or [])[:2]:
            print(f"   also:       {alt['name']}  [food_id {alt['food_id']} "
                  f"measure {alt['measure_id']}]")
    for e in result["failed"]:
        print(f"\n✗  {e['guess']['label']}: {e['error']}")
    return 0


def cmd_today(args) -> int:
    with CronometerClient() as client:
        data = client.get_daily_nutrition(args.date)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    c, t, r = data["consumed"], data["targets"], data["remaining"]
    print(f"{data['date']} — {data['entry_count']} entries\n")
    for key, unit in (("calories", "kcal"), ("protein_g", "g"), ("carbs_g", "g"), ("fat_g", "g")):
        left = f"  ({r[key]:+.0f} left)" if r.get(key) is not None else ""
        print(f"  {key:<10} {c[key]:>7.0f} / {t[key]:>7.0f} {unit}{left}")
    print()
    for e in data["entries"]:
        print(f"  [{e['meal']:<13}] {e['grams']:6.0f}g  {e['food_name']}  (id {e['entry_id']})")
    return 0


def cmd_add(args) -> int:
    with CronometerClient() as client:
        print(json.dumps(client.add_entry(args.food_id, args.measure_id, args.grams,
                                          date=args.date, diary_group=args.meal),
                         indent=2, default=str))
    return 0


def cmd_remove(args) -> int:
    with CronometerClient() as client:
        print(json.dumps(client.remove_entries(args.entry_ids, date=args.date),
                         indent=2, default=str))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="cli.py", description="crono-vision")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search foods and show how they'd be ranked")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--branded", action="store_true", help="prefer branded database entries")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("photo", help="analyze a photo and log it")
    s.add_argument("path")
    s.add_argument("--meal", help="breakfast/lunch/dinner/snacks (default: infer)")
    s.add_argument("--date", help="YYYY-MM-DD, 'today' or 'yesterday'")
    s.add_argument("--hint", help="free text, e.g. 'the chicken is 200g'")
    s.add_argument("--dry-run", action="store_true", help="analyze but don't write")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_photo)

    s = sub.add_parser("today", help="show the day's totals and entries")
    s.add_argument("date", nargs="?")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_today)

    s = sub.add_parser("add", help="log an entry directly")
    s.add_argument("food_id", type=int)
    s.add_argument("measure_id", type=int)
    s.add_argument("grams", type=float)
    s.add_argument("--meal", default="auto")
    s.add_argument("--date")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("remove", help="delete entries by id (see `today`)")
    s.add_argument("entry_ids", type=int, nargs="+")
    s.add_argument("--date")
    s.set_defaults(func=cmd_remove)

    args = p.parse_args()
    try:
        return args.func(args)
    except CronometerError as e:
        print(f"cronometer: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
