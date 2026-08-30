# crono-vision

Point your phone at a plate of food, and it lands in your Cronometer diary.

Photo → Gemini identifies the foods and estimates portions → each guess is
matched against Cronometer's database → confident matches get logged, unsure
ones get handed back to you. There's a CLI for local use and a serverless
endpoint for an iOS Shortcut.

Works on a free Cronometer account. No Gold subscription, no MCP runtime.

```
photo ──▶ vision.py ──▶ matcher.py ──▶ cronometer_client.py ──▶ your diary
          (Gemini)      (which food    (mobile.cronometer.com)
                         is it, really?)
                            │
                            └─▶ not sure? → needs_review, nothing written
```

## Layout

| File | What it does |
|---|---|
| [cronometer_client.py](cronometer_client.py) | The Cronometer API. Plain class, no framework. Auth, search, add/remove entries, daily nutrition. |
| [vision.py](vision.py) | Gemini call. Photo in, list of `{query, grams, confidence}` out. |
| [matcher.py](matcher.py) | Picks which search result is actually the food in the photo, and says when it isn't sure. |
| [pipeline.py](pipeline.py) | Wires the three together and writes a one-line summary. |
| [cli.py](cli.py) | Local driver — search, log a photo, check the day. |
| [api/index.py](api/index.py) | Vercel endpoint the camera page and the Shortcut both hit, at `/api`. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill it in — .env is gitignored
```

You need a Cronometer login and a [Gemini API key](https://aistudio.google.com/apikey).

**Set `CRONOMETER_TIMEZONE`.** It's not decoration: servers run in UTC, so
without it a 9pm dinner photo files itself under tomorrow.

## Using it locally

```bash
# See how the matcher ranks things — the fastest way to sanity-check a food
python cli.py search "greek yogurt"

# Analyze a photo without writing anything
python cli.py photo lunch.jpg --dry-run

# For real, with a nudge
python cli.py photo lunch.jpg --meal dinner --hint "the rice is about 200g"

python cli.py today
python cli.py remove 123456        # undo, ids come from `today`
```

`--hint` is worth using. Telling it "this is a 250g steak" beats any amount of
prompt tuning, because portion size is the part vision is genuinely bad at.

## Portions: read this before trying to fix them

The model does not measure portions. It recognises the food and reports what
that food *typically* weighs. Measured, on the same apple:

| | |
|---|---|
| Tiny in frame vs filling it | 170g vs 182g |
| 0.8× vs 2.0× a bank card in frame (~15× the volume) | 180g vs 180g |
| Same, forced to write the width in mm before the grams | measured 88mm vs 510mm → **still** 180g vs 220g |
| Same, on a thinking model (4.5-7.9s) | 182g vs 182g |

The third row is the one that settles it: told to measure against a reference,
the model *correctly perceives* the size difference, writes it down, and then
answers with the prior anyway. A thinking model does no better and costs five
times as long. This was attempted with prose scaffolding, with numeric
scaffolding, and with a bigger model; none of it moved the number, so please
don't spend an afternoon on prompt wording.

What that means in practice:

- **Whole standard items** — an apple, a banana, an egg — come out accurate,
  because the prior *is* the right answer.
- **Anything portioned** — rice, pasta, meat, cereal, anything in a bowl — comes
  out as a generic serving regardless of how much is actually there.

So the grams are a starting point, and there are two ways to move them. A
`hint` at capture time ("the rice is about 200g") overrides the prior before
anything is written and is the only thing that reliably does. After the fact,
the result card carries **portion chips** — ½× ¾× 1½× 2× — which re-log the
same food at a new weight, one tap, no keyboard. With several items on the
card, tap a row to choose which one the chips act on.

Under the hood a chip is just the swap endpoint with the food held constant
and the grams moving, so there's no extra endpoint and no new surface to
secure.

## Deploying

```bash
npm i -g vercel
vercel                              # first deploy, links the project
vercel env add CRONOMETER_EMAIL     # repeat for each var in .env.example
vercel --prod
```

Confirm it came up — the health check needs no auth and tells you which env
vars actually made it, **and which Gemini model is really live**:

```bash
curl https://<your-app>.vercel.app/api
```

```json
"model": "gemini-3.5-flash-lite",
"model_pinned_by_env": false
```

If `model_pinned_by_env` is `true`, a `GEMINI_MODEL` in your Vercel project is
outranking the default in [vision.py](vision.py) — worth knowing, because that
is invisible from the code and was once worth 40 seconds a photo.

`memory: 1024` in [vercel.json](vercel.json) is not about memory. Vercel scales
CPU with it, and this function is latency-bound, not memory-hungry — the larger
size roughly halves the cold-start import cost that otherwise lands on whoever
presses the shutter first. Billing is GB-seconds, so paying twice per second
for half as many seconds is close to a wash. Note that `vercel.json` rejects
unknown keys, comment fields included, so that reasoning has to live here.

`maxDuration` is the ceiling the vision budget in [vision.py](vision.py) is
sized against. If you raise one, raise the other.

## The iOS Shortcut

Four actions:

1. **Take Photo** (or *Select Photos*)
2. **Get Contents of URL**
   - URL: `https://<your-app>.vercel.app/api?meal=auto`
   - Method: `POST`
   - Headers: `Authorization` → `Bearer <your CRONO_VISION_TOKEN>`
   - Request Body: **File** → the photo from step 1
3. **Get Dictionary Value** — key `summary`
4. **Show Notification** (or *Speak Text*)

Add it to your Home Screen or Action Button. That's the whole loop.

Query params, all optional: `meal` (auto/breakfast/lunch/dinner/snacks),
`date` (`YYYY-MM-DD`, `today`, `yesterday`), `hint`, `dry_run`.

For a hint you can type at capture time, insert an **Ask for Input** action and
append it as `&hint=[input]`. Or send JSON instead of a file:

```json
{"image": "<base64>", "meal": "lunch", "hint": "large portion"}
```

Set `dry_run=true` on the URL for the first few runs.

## How a capture flows

A plain `POST /api` runs the whole pipeline and answers once. That's what
the Shortcut wants — one request, one notification — and it's what `log_photo`
does.

The camera page splits the same work into two requests, because it has a
screen to keep honest:

| | |
|---|---|
| `POST /api?phase=analyze` | Gemini, then a database search per food. Writes nothing. Returns `pending` — the items it would log, each carrying the `plan` to write. |
| `POST /api?action=commit` | Writes those plans. Body is `{date, meal, items}` — hand back the `pending` list as-is. |

The split buys two things. The detection markers can appear the moment the
analysis lands, instead of after the diary writes, which is most of the
stare-at-a-frozen-frame time. And a retry becomes safe: `commit` re-sends the
grams `analyze` already settled on, so the duplicate check matches. Retrying
the combined call re-ran vision, and a fresh estimate of 148g against a
logged 150g slid straight past that check and logged lunch twice.

Latency is mostly Gemini, and **which model you point it at matters more than
everything else here combined.** Measured on one photo:

| | |
|---|---|
| `gemini-3.5-flash-lite` (default) | ~1.8s |
| `gemini-3.6-flash` | ~9.4s |
| `gemini-3.7-flash` | ~6-8s, first to start returning 429 |

Same items, same boxes, same portions. The big models spend those seconds
*thinking* — they report hundreds of thinking tokens where the Lite models
report none — and reach the same answer, because identifying an apple was
never a reasoning problem. **Don't set `GEMINI_MODEL` to something bigger
because it sounds better.** That one env var was worth 5-9x here, and pinning
a thinking model is also how a capture becomes a 40-second wait: the fallback
chain fires on a 429, and a refusal isn't fast either — an overloaded endpoint
took a measured 17.6s just to answer 503.

Hence the caps in [vision.py](vision.py): 12s per attempt, 30s across all of
them, fallbacks ordered fastest-first. A fallback fires when the person has
*already* been waiting, which is the worst possible moment to reach for the
slowest model in the list.

The rest is kept out of the way: the Cronometer login runs *during* the Gemini
call rather than after it, a batch of writes reads the day once instead of once
per food, and the writes go out together. On a four-item plate at a 120ms round
trip that's about 2.0s of non-Gemini overhead down to about 0.5s.

Nothing in the browser runs without a deadline (`ANALYZE_TIMEOUT_MS` and
friends in [index.html](index.html)), and Gemini's model-fallback chain has a
total `budget`, not just a per-attempt `timeout` — four models at 60s each
inside a function Vercel kills at 60s is how a slow Gemini used to end as a
504 *after* the writes had landed, which looked from the phone like nothing
happened and from Cronometer like everything did.

## Which meal it lands in

Three layers, in strict order of authority:

1. **An explicit `meal`** wins outright. Nothing overrides it.
2. **The clock** alone picks breakfast / lunch / dinner.
3. **The food** may only answer one question — *is this a snack?* — and if so
   it moves to snacks, wherever the clock had put it.

Defaults, overridable with `CRONO_MEAL_WINDOWS`:

| | |
|---|---|
| 04:00 – 10:29 | breakfast |
| 10:30 – 14:59 | lunch |
| 15:00 – 20:59 | dinner |
| 21:00 – 03:59 | snacks |

The day starts at 04:00, not midnight, so a 1am plate is the tail of last
night rather than tomorrow's breakfast.

Layer 3 is deliberately one-directional, and that's the whole design. Letting
the model pick the slot outright was tried and reverted: a steak photographed
at 12:27am came back as *lunch*, because the plate read as a lunch-type meal.
It has no vocabulary for a time slot now — the schema offers it `meal`,
`snack`, `unknown` and nothing else — so the only move it can make is pulling
a bowl of strawberries at noon out of lunch and into snacks, which is the case
the clock genuinely gets wrong. It can't promote in the other direction
either: a steak at 3am stays in snacks rather than becoming "dinner", because
at that hour the slot is the honest answer.

A model that answers with an old-style `"lunch"` anyway is parsed as
`unknown` and ignored, so a stale deployment degrades to clock-only rather
than back to the bug.

Meal grouping is organizational, not nutritional — Cronometer totals
everything per day regardless. Pass `meal=uncategorized` if you'd rather it
not guess at all.

## What gets logged, and what doesn't

Nothing is written unless three things hold: Gemini was reasonably sure what
the food is, the top database match scores above threshold, and it's clearly
ahead of the runner-up. Anything else comes back under `needs_review` with the
alternatives attached, because fixing a wrong Cronometer entry is more annoying
than adding a missing one.

Tune the thresholds in [matcher.py](matcher.py) (`MIN_CONFIDENCE`,
`MIN_MARGIN`) and [pipeline.py](pipeline.py) (`MIN_VISION_CONFIDENCE`).

The ranking blends three signals — Cronometer's own relevance score (40%),
token F1 against the food name (45%), and how much to trust the source
database (15%). The middle one is what stops "Chicken Breast Nuggets, Breaded,
Frozen" from beating plain grilled chicken: extra words you didn't ask for cost
precision. Source weighting prefers generic USDA/NCCDB entries for a plate of
food, and flips to branded when Gemini reports legible packaging.

## Tests

Both suites stub the network. No credentials, nothing written anywhere.

```bash
python test_offline.py     # ranking maths, date handling, response parsers
python test_endtoend.py    # full photo→diary path, plus the HTTP handler
```

## Notes on the API

`mobile.cronometer.com` is what the free Android app talks to. Two protocols
share it: `/api/v2/*` is JSON-RPC-ish (everything POSTs, the session rides in
an `auth` block inside the body), while `/api/v3/*` is real REST with the
session in an `x-crono-session` header. Deletes are v3, everything else here is
v2. Login returns HTTP 200 with `result: FAIL` when your password is wrong,
which is worth knowing before you debug it.

Sessions are cached to the system temp dir for 12h. On Vercel that's `/tmp`,
shared between warm invocations — which is the difference between one login a
day and getting locked out with "Too Many Attempts."

It's an undocumented API. It can change without notice.
