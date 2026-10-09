# Prompt for a redesign of the results page

Paste everything below the line into a fresh Claude conversation, with a
screenshot of `docs/index.html` attached. The complaint is visual; a screenshot
carries it far better than this description does.

Regenerate first so the screenshot is current:

```
python scripts/build_report.py
```

---

I need you to redesign a results page for a machine-learning project. Give me
back a single HTML file. I will wire the numbers back in myself.

## Who reads it and for how long

A hiring manager, over a shared screen, for about twenty seconds, while I talk
over it. It is not a dashboard anyone returns to — it is the "so what" of a
project walkthrough. If they cannot tell what the finding is without me
narrating, the page has failed.

Technical reader, reading for clarity rather than depth.

## The project

`Arrival` predicts delivery ETAs. It sits on top of a data pipeline called
Dispatch and its point is not the model — it is the **feature store**: features
are declared once and retrieved two ways, point-in-time for training and by
entity for serving, and a test asserts the two paths return identical values.

## The finding the page exists to deliver

A **leaked feature window** — one that filters on when a trip was *assigned*
rather than when it *finished* — makes a model look better than it is, and the
size of that flattery depends on how much history there is:

```
                        19,312 trips       4,809 trips
honest lift over
the flat promise             2.830             3.065     minutes of MAE
in-flight leak gap           0.924             3.746     minutes of MAE
```

On the small dataset the leak appears to buy **more than the entire real
improvement**. On four times the history it buys about a third of it. Nothing
about the leak changed — the honest features simply got enough history to be
nearly as informative, so the stolen information stopped being worth much.

**That contrast is the page.** A leak flatters hardest exactly when a project is
small, new, and least likely to be checked.

There is a second finding, currently given equal weight: a model that breaks at
09:00 is **invisible on an accuracy dashboard until its labels arrive**, because
a trip's true duration is only known ~35 minutes after it was predicted.

## What the page looks like right now

1. **Header** — a three-line headline carrying two comparisons, then a stamp
   ("generated … from 19,312 trips … every figure computed at build time").
2. **"What each model scores on the held-out trips"** — four horizontal bars of
   MAE, one per model (flat promise, honest, leaky in-flight, leaky no-window).
   Bar length is MAE. Two bars are orange to mark the models that cheated. A
   caveat paragraph follows.
3. **"A model that broke at 09:00"** — a six-column table: time, trips scored,
   MAE healthy, MAE broken, a verdict word (identical / diverged), labels
   pending. Three rows: 09:20, 09:40, 10:00.
4. **"Serving"** — four KPI tiles: p50 10ms, p99 2000ms, mean 109ms, 1,000
   requests.
5. **"Could the experiment have seen it?"** — three KPI tiles: real effect,
   measured, smallest detectable; plus a paragraph.
6. **"How it is built"** — prose, plus links to two architecture diagrams.

## What is wrong with it

The part I most want your judgement on. My own read:

- **The bar chart rewards the thing the page is warning about.** Bars are MAE,
  where *lower is better*, so the cheating model has the shortest bar and
  visually wins the chart. Colour is the only thing saying it is the bad one, and
  colour is the weakest signal on the page.
- **The headline carries two comparisons across three lines.** Twenty seconds
  does not survive "A is worth X, B is worth Y, and on a quarter of the data they
  were P and Q."
- **The strongest finding is not on the page.** The 19,312-versus-4,809 contrast
  — the one that explains *why* the leak matters and when — exists only in the
  README. The page shows one dataset and asserts the rest.
- **Five sections of roughly equal weight** means none of them is the headline.
  Serving latency gets the same visual budget as the central result.
- **The KPI tiles look like a SaaS dashboard**, which is the wrong register for a
  result that is about being careful.

Tell me if you disagree. I am not attached to the bars, the tiles, the section
order, or the section count.

## What I want

The leak contrast legible in **under five seconds**, with the history-size
comparison visible on the same screen at 1440px. The outage finding can be the
clear second thing. Serving and experiment detail can sit below the fold.

Think about what form suits "the same mistake is worth a lot here and almost
nothing there". Four bars of a lower-is-better metric is one answer. It is
probably not the best one.

## Hard constraints

- **One self-contained HTML file.** No build step, no framework, no npm.
- **No external requests at all.** No CDN, no web fonts, no remote images. It is
  served from GitHub Pages and must render with the network off.
- **Inline SVG only** for charts. No charting library.
- **Light and dark mode**, both deliberate. Use `prefers-color-scheme` and also
  honour `data-theme="dark"` and `data-theme="light"` on `:root`.
- **Phone width, no horizontal scrolling.**
- **Colour must never be the only carrier of meaning.** Anything identified by
  colour needs a direct label beside it. Assume a colourblind reader and a
  greyscale print — this matters more here than usual, because colour is
  currently the only thing distinguishing an honest model from a cheating one.
- **Nothing important may be hover-only.**
- **Lower-is-better must be legible as such.** If you keep a length-based
  encoding for MAE, make the direction unmistakable without reading the caption.

## How the numbers get in

The page is built by a Python f-string in `scripts/build_report.py`, so **every
literal `{` and `}` in your CSS will need doubling** when I paste it in. Write
normal CSS; I will handle the escaping. Just keep the structure regular and tell
me clearly which values are injected.

Values currently available, with the figures they held on the last run:

```
honest lift          2.830        leak gap, in-flight   0.924
leak gap, no window  0.086        trips                 19,312
train / test         15,449 / 3,863
per model: mae, rmse, p50_error, p90_error, breach_rate, n
   flat promise  11.019 / 25.636 p90 / 0.719 breach
   honest         8.190 / 16.430 / 0.410
   leaky inflight 7.265 / 15.806 / 0.464
   leaky nowindow 8.103 / 16.345 / 0.422
outage rows: 09:20 (46 scored, healthy 0.500, broken 0.500, 154 pending)
             09:40 (66, 0.500, 3.636, 134)
             10:00 (86, 0.500, 10.930, 114)
latency: n 1000, p50 10ms, p99 2000ms, mean 109ms
experiment: real effect 0.16 min, measured 0.31, detectable 1.10, 800 trips
```

If your design needs a figure that is not in that list, say so and I will compute
it — do not invent one.

## Things not to do

- No logo, no invented company name, no nav bar.
- Do not make it look like a SaaS analytics product.
- **Do not drop the caveat.** The trips are synthetic where Dispatch's warehouse
  is absent, and the page must say so under the chart it qualifies.
- Do not present the leaky models as an ablation study or a tuning result. They
  are mistakes, deliberately made, and the page must read that way.
- No animation beyond a hover state.

## Palette (keep unless you have a reason)

```
light surface  #fcfcfb      dark surface  #1a1a19
light text     #0b0b0b      dark text     #ffffff
honest         #2a78d6      dark          #3987e5
leaked         #eb6834      dark          #d95926
```

Those two were checked for colourblind separation and contrast in both modes. If
you change them, say what you checked.
