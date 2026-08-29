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

## Deploying

```bash
npm i -g vercel
vercel                              # first deploy, links the project
vercel env add CRONOMETER_EMAIL     # repeat for each var in .env.example
vercel --prod
```

Confirm it came up — the health check needs no auth and tells you which env
vars actually made it:

```bash
curl https://<your-app>.vercel.app/api
```

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

Latency is mostly Gemini and always will be. The rest is kept out of the way:
the Cronometer login runs *during* the Gemini call rather than after it, a
batch of writes reads the day once instead of once per food, and the writes go
out together. On a four-item plate at a 120ms round trip that's about 2.0s of
non-Gemini overhead down to about 0.5s.

Nothing in the browser runs without a deadline (`ANALYZE_TIMEOUT_MS` and
friends in [index.html](index.html)), and Gemini's model-fallback chain has a
total `budget`, not just a per-attempt `timeout` — four models at 60s each
inside a function Vercel kills at 60s is how a slow Gemini used to end as a
504 *after* the writes had landed, which looked from the phone like nothing
happened and from Cronometer like everything did.

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
