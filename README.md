# Arrival

An ETA prediction service with a feature store, built on top of
[Dispatch](https://github.com/JOGURNAUT/dispatch): **point-in-time features →
training → serving → monitoring → A/B**, with **107 tests** and a generated
results page.

**[Live results →](https://jogurnaut.github.io/arrival/)** — every figure
computed when the page is built.
**[Architecture →](https://jogurnaut.github.io/arrival/architecture.html)** ·
**[What Dispatch hands Arrival →](https://jogurnaut.github.io/arrival/handoff.html)**

```
A leaked feature window buys 0.92 minutes of MAE that does not exist.
The honest model is worth 2.83 — and on a quarter of the data,
the same leak was worth 3.75 while the honest model was worth 3.07.
```

That is the finding, and the second line is the important half. Filtering a
feature window on when a trip was *assigned* rather than when it *finished*
flatters a model — and it flatters it hardest when there is least history to
learn from, which is exactly when a model is being prototyped and exactly when
nobody is checking. None of it survives production, and nothing in the training
run says so.

## Run it

Nothing to sign up for. No GPU, no warehouse account, no API key.

```bash
pip install -r requirements.txt

python -m arrival.train          # train every model, print the comparison
python scripts/build_report.py   # write docs/index.html
python -m pytest -q              # 107 tests

streamlit run dashboard/app.py   # optional
```

It reads Dispatch's warehouse if it finds one, and falls back to a seeded
generator so a fresh clone works on its own.

## Why this sits on top of Dispatch

Dispatch is the pipeline that lands trip telemetry and measures how wrong a flat
delivery promise is. Arrival predicts the number that promise should have been.
Its gold table `fct_trip` is already a training table, and two of its columns are
what make this project possible:

| | |
|---|---|
| `assigned_at` | the instant a prediction was due. Nothing after it was knowable |
| `tat_minutes` | the label, which only exists ~35 minutes later |

The first defines what a feature is allowed to see. The second is why monitoring
cannot simply compare predictions against truth.

That is the same shape as a real ML platform: a warehouse holds the history, a
feature store serves it to a model, and an inference service answers in
milliseconds.

## A feature has a value only at a moment in time

"The driver's average trip time" is not a number. It is a number **as of** some
instant, and a different number a minute later.

Training asks for it as of when each historical trip was assigned. Serving asks
for it as of now. If those two questions are answered by two pieces of code, they
drift — and the model is scored offline on numbers it will never be served.

So every retrieval carries an `as_of`, both paths are driven by the same
declarations, and a test replays a day asserting they agree:

```python
# tests/test_parity.py
assert store.online(target, as_of).values == offline[target.trip_id]
```

Six hundred trips, exact dictionary equality, each at its own `assigned_at` with
finished trips pushed in as they complete. One trip agreeing is a coincidence.

## The leak, measured

Trained on Dispatch's warehouse — 19,312 completed trips, 15,449 train, 3,863
test, 60 drivers with a median of 320 trips each:

```
model                        MAE     p90 error   breach rate
flat promise               11.019      25.636         0.719
ridge, point-in-time        8.190      16.430         0.410
ridge, leaky: in-flight     7.265      15.806         0.464
ridge, leaky: no window     8.103      16.345         0.422
```

**The honest model takes 2.830 minutes off the flat promise.** That is the
improvement that exists.

**The in-flight leak looks 0.924 minutes better again.** Its window filters on
`assigned_at`, so it reads trips that had not finished when the estimate was due
— including the trip being predicted, whose own duration enters its own feature.

**The windowless variant gaps 0.086 minutes**: small enough to shrug at, wrong
for exactly the same reason, and the shape a `GROUP BY` written without a date
filter actually takes. A leak that announces itself gets caught. This is what
the other kind looks like.

### The leak is largest when the history is thinnest

The same comparison on a quarter of the data — 4,809 trips, about 80 per driver
rather than 320:

```
                      4,809 trips      19,312 trips
honest lift               3.065             2.830
in-flight leak gap        3.746             0.924
```

On the smaller set the leak appears to buy *more than the entire real
improvement*. On the larger one it buys a third of it. Nothing about the leak
changed; the honest features simply got enough history to be nearly as
informative, so the stolen information stopped being worth much.

Which is the uncomfortable part. The leak flatters hardest exactly when a
project is small, new, and least likely to be checked.

## A model that broke at 09:00

Every estimate doubled from 09:00. Trips take 35 minutes, so a trip's duration is
known 35 minutes after it was predicted.

```
time     trips scored    MAE      labels pending
09:20          46       0.500          154
09:40          66       3.636          134
10:00          86      10.930          114
```

Twenty minutes into the outage the dashboard reads **the same MAE as a model
that never broke** — every trip it could score was predicted before 09:00.
Accuracy is structurally behind by the label delay, and no amount of refreshing
fixes it.

Prediction drift is not behind. Predictions exist the moment they are served, and
the broken model's distribution has already moved: **PSI 10.3 against 0.0 for the
healthy model**, thirty-five minutes before accuracy registers anything.

## Could the experiment have seen it?

A treatment that genuinely is 2% better — 0.16 minutes off an 8 minute MAE — run
across 800 drivers:

```
real effect                0.16 min
measured difference        0.31 min
smallest detectable        1.10 min     ->  inconclusive
needed per arm         22,075 trips
```

The measured difference is larger than the real one and smaller than what the
experiment could resolve. Without the third number, the second goes into a deck
as a win.

Assignment hashes a stable entity id rather than being drawn per request —
per-request randomness puts the same driver in both arms across their shift,
which does not add noise to the comparison so much as remove the comparison.

## Serving

```
p50     10 ms
p99   2000 ms
mean   109 ms
```

The mean would clear any alarm anyone would set while one request in twenty takes
two seconds. Latency is reported as percentiles, never a mean.

A service asked about a driver it has never seen still answers, with the flat
promise — and marks the answer `fallback`, because "the model said 23 minutes"
and "nobody could tell us anything" are the same number and completely different
facts. The fallback rate is what separates *the model is live* from *the model
has been failing open since Tuesday*.

## Tests

**107, none of which need a network, an account or a GPU.**

| | |
|---|---|
| `test_parity.py` | 11 — the offline and online paths return identical values, including a replayed day and out-of-order arrival |
| `test_features.py` | 18 — windows, cold start, `LeakageError`, and a check against a deliberately quadratic reference implementation |
| `test_training.py` | 19 — chronological split, the ridge solver, registry round-trips |
| `test_serve.py` | 22 — percentiles, fallback flagging, health |
| `test_monitor.py` | 16 — PSI, drift, and the blind window |
| `test_experiment.py` | 21 — assignment stability, guardrails, MDE |

Two of them exist because they caught real bugs during the build:

- **PSI binning used `bisect_right`**, so a feature that collapsed to a single
  constant — a pipeline returning its default for every entity, the most total
  drift there is — scored `0.0`, meaning *no drift*. The most broken case read as
  the healthiest.
- **Registry timestamps were second-precision**, so two models saved in the same
  second tied and `latest()` resolved by hash instead of by time.

## Layout

```
arrival/
  contracts.py     the types and Protocols everything codes against
  source.py        Dispatch's warehouse, or a seeded stand-in
  features.py      one declaration, two retrieval paths
  train.py         ridge in pure stdlib, chronological split, leak variants
  registry.py      content-hashed model artifacts
  serve.py         PredictionService, percentiles, flagged fallbacks
  monitor.py       PSI, prediction drift, lagged-label accuracy
  experiment.py    hashed assignment, guardrails, MDE
tests/             107
scripts/           build the report
dashboard/         Streamlit
docs/index.html    generated
```

## Honest scope

The trips are synthetic where Dispatch's warehouse is absent, so the absolute
minutes are a demonstration rather than a measurement. What is not circular is
the ordering: a leak flatters a model, the flattery does not survive serving, and
an accuracy dashboard cannot see an outage until its labels arrive.

The model is ridge regression in pure Python. Gradient boosting would score
better and would teach nothing this project is about — the subject is the
platform around the model, not the model.
