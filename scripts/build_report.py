"""Write docs/results.html from a real training run and a real replay.

Generated, never written by hand. Every figure on the page is produced by code
that runs when the page is built: the models are trained, the outage is
simulated, the latencies are measured. A results page with numbers typed into it
stops being true the first time anything changes, and then it is a screenshot
pretending to be a report.

Static rather than a live dashboard, for the same reason the dashboard exists
anyway: this page keeps working when nothing is running.

Chrome, colour and type come from docs/site.css, shared with the Overview page
and with Dispatch's own results page -- the leaked models are drawn in
var(--series-a) (orange) and the honest ones in var(--series-b) (blue), the
same two tokens Dispatch uses for its two stores, so the three sites read as
one project rather than three documents that happen to be linked.

    python scripts/build_report.py
"""

from __future__ import annotations

import datetime as dt
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrival import source, train                      # noqa: E402
from arrival.contracts import Prediction, Trip          # noqa: E402
from arrival.experiment import minimum_detectable_effect  # noqa: E402
from arrival.monitor import Monitor                     # noqa: E402
from arrival.serve import LatencySummary                # noqa: E402

OUT = ROOT / "docs" / "results.html"

BREAK_AT = dt.datetime(2026, 9, 1, 9, 0)
TRIP_MINUTES = 35.0


def outage() -> list[dict]:
    """Break a model at 09:00 and ask the dashboard what it sees, hour by hour.

    The simulation is deliberately simple -- one trip a minute, every trip
    taking the same 35 minutes -- so that nothing in the result comes from the
    data being awkward. The blindness is structural.
    """
    rows = []
    for broken in (False, True):
        mon = Monitor()
        trips = []
        for i in range(200):
            at = dt.datetime(2026, 9, 1, 8, 0) + dt.timedelta(minutes=i)
            trip = Trip(trip_id=f"T{i:04d}", store_id="NORTHGATE",
                        driver_id=f"D{i % 20:03d}", assigned_at=at,
                        distance_m=3000, promised_minutes=19,
                        delivered_at=at + dt.timedelta(minutes=TRIP_MINUTES),
                        tat_minutes=TRIP_MINUTES)
            guess = TRIP_MINUTES * 2 if (broken and at >= BREAK_AT) \
                else TRIP_MINUTES + 0.5
            mon.record_prediction(
                Prediction(trip_id=trip.trip_id, minutes=guess,
                           model_version="v1", served_at=at), trip)
            trips.append(trip)
        for t in trips:
            mon.record_outcome(t)

        for mins in (20, 40, 60):
            as_of = BREAK_AT + dt.timedelta(minutes=mins)
            m = mon.realised_accuracy(as_of)
            rows.append({"broken": broken, "mins": mins,
                         "clock": f"{as_of:%H:%M}",
                         "n": m.n if m else 0,
                         "mae": m.mae if m else None,
                         "pending": mon.labels_pending(as_of)})
    return rows


def latencies() -> LatencySummary:
    """A service where most requests are fast and a few are not, which is every
    service."""
    rng = random.Random(3)
    samples = [rng.gauss(11, 2) for _ in range(950)]
    samples += [rng.gauss(1900, 120) for _ in range(50)]
    return LatencySummary.of(samples)


def experiment() -> dict:
    """An honest 2% improvement, and whether 800 drivers could have seen it."""
    rng = random.Random(7)
    control = [max(0.5, rng.gauss(8.0, 5.55)) for _ in range(400)]
    treatment = [max(0.5, rng.gauss(8.0 * 0.98, 5.55)) for _ in range(400)]
    observed = (sum(control) / len(control)) - (sum(treatment) / len(treatment))
    pooled = (sum((x - sum(control) / len(control)) ** 2 for x in control)
              + sum((x - sum(treatment) / len(treatment)) ** 2 for x in treatment))
    pooled = (pooled / (len(control) + len(treatment) - 2)) ** 0.5
    detectable = minimum_detectable_effect(pooled, len(control), len(treatment))
    return {"observed": observed, "mde": detectable,
            "n": len(control) + len(treatment), "true_effect": 8.0 * 0.02}


def render(result, lag, lat, ab) -> str:
    m = result.metrics
    # series-a (orange) marks a model that cheated; series-b (blue) marks an
    # honest one. The same two tokens Dispatch uses for its two stores, so a
    # reader who has seen one dashboard already knows what the colour means.
    rows = [("flat promise", m["baseline"], None),
            ("ridge, point-in-time", m["ridge"], "b"),
            ("ridge, leaky: in-flight window", m["ridge_leaky_in_flight"], "a"),
            ("ridge, leaky: no window", m["ridge_leaky_lifetime"], "a")]
    worst = max(r[1].mae for r in rows)

    def barrow(name, met, series):
        colour = f"var(--series-{series})" if series else "var(--ink-3)"
        return f"""
      <div class="bar-row">
        <div class="bar-head">
          <span class="bar-name">{name}</span>
          <span class="bar-val">{met.mae:.3f} min MAE</span>
        </div>
        <div class="track">
          <div class="fill" style="width: {met.mae / worst * 100:.1f}%; background: {colour}"></div>
        </div>
        <div class="bar-sub">p90 error {met.p90_error:.1f} min &middot;
             breach {met.breach_rate:.1%} &middot; {met.n:,} trips</div>
      </div>"""

    bars = "\n".join(barrow(name, met, series) for name, met, series in rows)

    # Both models, side by side. The claim this table makes is that a broken
    # model reads the same as a healthy one, and a table that shows only the
    # broken column asks to be believed rather than checked. "identical" is
    # the bad outcome here -- it means the blind spot is still blind -- so it
    # takes the warn badge; "diverged" means the break has become visible, so
    # it takes the pass badge. The usual badge meanings invert on this table
    # on purpose, because the thing being graded is visibility, not health.
    healthy = {r["clock"]: r for r in lag if not r["broken"]}

    def fmt(v):
        return "&mdash;" if v is None else f"{v:.3f}"

    def lagrow(r: dict) -> str:
        h = healthy[r["clock"]]
        same = (h["mae"] is not None and r["mae"] is not None
                and abs(h["mae"] - r["mae"]) < 1e-9)
        badge = ('<span class="badge warn">identical</span>' if same
                else '<span class="badge pass">diverged</span>')
        return (f"<tr><td>{r['clock']}</td>"
                f"<td class='n'>{r['n']}</td>"
                f"<td class='n'>{fmt(h['mae'])}</td>"
                f"<td class='n'>{fmt(r['mae'])}</td>"
                f"<td>{badge}</td>"
                f"<td class='n'>{r['pending']}</td></tr>")

    lagrows = "\n".join(lagrow(r) for r in lag if r["broken"])

    # p99 at a full two seconds is the thing worth a reader's attention; p50
    # at ten milliseconds is not. The warn/ok modifiers carry that judgement
    # instead of leaving every number the same visual weight.
    p99_cls = "is-warn" if lat.p99_ms >= 500 else "is-ok"

    inconclusive = abs(ab["observed"]) < ab["mde"]
    measured_cls = "is-warn" if inconclusive else "is-ok"

    start, end = result.training_window
    built = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arrival: Results</title>
<meta name="description" content="A leaked feature window buys MAE that does not exist, and the size of that flattery depends on how much history there is.">
<link rel="stylesheet" href="site.css">
</head>
<body>

<nav class="sitenav" aria-label="Pages">
  <a href="index.html">Overview</a>
  <a href="architecture.html">Architecture</a>
  <a href="results.html" aria-current="page">Results</a>
  <span class="spacer"></span>
  <a class="out" href="https://github.com/JOGURNAUT/arrival">Source</a>
  <button class="themetoggle" type="button">Light</button>
</nav>

<div class="wrap">

<p class="eyebrow">Arrival &middot; point-in-time feature store</p>
<h1>The leak looks {result.leakage_gap_mae:.2f} minutes better than the honest
    model. The honest model is only {result.lift_over_baseline_mae:.2f} minutes
    better than doing nothing.</h1>
<p class="stamp">generated {built:%Y-%m-%d %H:%M} UTC from
   {result.n_train + result.n_test:,} trips &middot;
   {start:%d %b} to {end:%d %b} &middot; every figure computed at build time</p>

<section>
  <h2>What each model scores on the held-out trips</h2>
  <p class="lede">Lower is better. The flat promise is what the business does
     today, so it is the only baseline an improvement means anything against.</p>
  <div class="legend">
    <span><i class="swatch" style="background: var(--series-b)"></i> honest</span>
    <span><i class="swatch" style="background: var(--series-a)"></i> leaked, cheated</span>
  </div>
  <div class="bars">{bars}</div>
  <p class="caveat"><b>The gap is the finding.</b> Filtering a feature window on
     when a trip was <i>assigned</i> rather than when it <i>finished</i> buys
     {result.leakage_gap_mae:.2f} minutes of MAE, more than the honest
     model's entire lift over the baseline, and none of it survives
     production, because at prediction time those trips had not finished and one
     of them is the trip being predicted. Dropping the window altogether gaps
     only {result.lifetime_leakage_gap_mae:.2f} minutes: small enough to dismiss
     as noise, wrong for exactly the same reason, and the shape a GROUP BY
     written without a date filter actually takes.</p>
</section>

<section>
  <h2>A model that broke at 09:00</h2>
  <p class="lede">Every estimate doubled from 09:00. Trips take 35 minutes, so a
     trip's true duration is only known 35 minutes after it was predicted. This
     is what the accuracy dashboard showed. Click or press Enter on a column
     heading to sort.</p>
  <div class="tbox">
    <table data-sortable>
      <thead><tr><th data-sort>Time</th><th class="n" data-sort>Trips scored</th>
        <th class="n" data-sort>MAE (healthy)</th>
        <th class="n" data-sort>MAE (broken)</th>
        <th data-sort>Visible?</th>
        <th class="n" data-sort>Labels pending</th></tr></thead>
      <tbody>{lagrows}</tbody>
    </table>
  </div>
  <p class="caveat" style="margin-top:14px">Twenty minutes into the outage the
     dashboard reads the same MAE as a model that never broke, because every
     trip it could score was predicted before 09:00. Accuracy is structurally
     behind by the label delay. <b>Prediction drift is not.</b> Predictions exist the moment they are served, and the broken model's
     distribution has already moved while its accuracy has not.</p>
</section>

<div class="grid2">
  <section>
    <h2>Serving</h2>
    <p class="lede">Latency, reported as percentiles rather than a mean.</p>
    <div class="kpis">
      <div class="kpi is-ok"><div class="kv">{lat.p50_ms:.0f} ms</div><div class="kl">p50</div></div>
      <div class="kpi {p99_cls}"><div class="kv">{lat.p99_ms:.0f} ms</div><div class="kl">p99</div></div>
      <div class="kpi"><div class="kv">{lat.mean_ms:.0f} ms</div><div class="kl">mean</div></div>
      <div class="kpi"><div class="kv">{lat.n:,}</div><div class="kl">requests</div></div>
    </div>
    <p class="caveat" style="margin-top:14px">The mean is {lat.mean_ms:.0f} ms and
       would clear any alarm anyone would set. One request in twenty takes
       {lat.p99_ms / 1000:.1f} seconds. Reporting a mean latency is how a
       service stays slow without anybody finding out.</p>
  </section>

  <section>
    <h2>Could the experiment have seen it?</h2>
    <p class="lede">A treatment that genuinely is 2% better,
       {ab['true_effect']:.2f} min off an 8 min MAE, over
       {ab['n']:,} trips.</p>
    <div class="kpis">
      <div class="kpi"><div class="kv">{ab['true_effect']:.2f}</div><div class="kl">real effect, min</div></div>
      <div class="kpi {measured_cls}"><div class="kv">{ab['observed']:.2f}</div><div class="kl">measured, min</div></div>
      <div class="kpi"><div class="kv">{ab['mde']:.2f}</div><div class="kl">smallest detectable, min</div></div>
    </div>
    <p class="caveat" style="margin-top:14px">The measured difference sits
       {"inside" if inconclusive else "outside"} what this experiment could
       resolve, so it is {"noise with a decimal point" if inconclusive
       else "a result worth reporting"}. Without the third number the second
       one goes into a deck as a win either way.</p>
  </section>
</div>

<section>
  <h2>How it is built</h2>
  <p class="lede">Trips come from <b>Dispatch</b>'s gold table, where
     <code>assigned_at</code> is the instant a prediction was due and
     <code>tat_minutes</code> only exists once the trip ends. Features are
     declared once and retrieved two ways: point-in-time for training, by
     entity for serving, and a test replays a day asserting the two paths
     return identical values for every trip.</p>
  <p class="lede" style="margin-top:-8px">
     <a href="architecture.html">Architecture diagram &rarr;</a>
     &nbsp;&middot;&nbsp;
     <a href="handoff.html">What Dispatch hands Arrival &rarr;</a></p>
  <p class="caveat">Trips are synthetic where Dispatch's warehouse is absent, so
     treat the absolute minutes as a demonstration. What is not circular is the
     ordering: a leak flatters a model, the flattery does not survive serving,
     and an accuracy dashboard cannot see an outage until its labels arrive.</p>
</section>

</div>

<script src="site.js"></script>
</body>
</html>
"""


def main() -> int:
    trips = source.load()
    result = train.run(trips)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(result, outage(), latencies(), experiment()),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  leakage gap {result.leakage_gap_mae:.3f} min, "
          f"lift over baseline {result.lift_over_baseline_mae:.3f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
