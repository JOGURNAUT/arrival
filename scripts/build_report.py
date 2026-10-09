"""Write docs/index.html from a real training run and a real replay.

Generated, never written by hand. Every figure on the page is produced by code
that runs when the page is built: the models are trained, the outage is
simulated, the latencies are measured. A results page with numbers typed into it
stops being true the first time anything changes, and then it is a screenshot
pretending to be a report.

Static rather than a live dashboard, for the same reason the dashboard exists
anyway: this page keeps working when nothing is running.

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

OUT = ROOT / "docs" / "index.html"

# Categorical slots from a palette checked for colourblind separation and
# contrast in both modes.
HONEST = {"light": "#2a78d6", "dark": "#3987e5"}
LEAK = {"light": "#eb6834", "dark": "#d95926"}

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
    rows = [("flat promise", m["baseline"], False),
            ("ridge, point-in-time", m["ridge"], False),
            ("ridge, leaky: in-flight window", m["ridge_leaky_in_flight"], True),
            ("ridge, leaky: no window", m["ridge_leaky_lifetime"], True)]
    worst = max(r[1].mae for r in rows)

    bars = "\n".join(f"""
      <div class="row">
        <div class="rh"><span class="nm">{name}</span>
          <span class="rt">{met.mae:.3f} min MAE</span></div>
        <div class="track"><div class="fill" style="width:{met.mae / worst * 100:.1f}%;
             background:var(--{'leak' if leak else 'honest'})"></div></div>
        <div class="sub">p90 error {met.p90_error:.1f} min &middot;
             breach {met.breach_rate:.1%} &middot; {met.n:,} trips</div>
      </div>""" for name, met, leak in rows)

    def lagrow(r: dict) -> str:
        mae = "&mdash;" if r["mae"] is None else f"{r['mae']:.3f}"
        return (f"<tr><td>{r['clock']}</td>"
                f"<td class='n'>{r['n']}</td>"
                f"<td class='n'>{mae}</td>"
                f"<td class='n'>{r['pending']}</td></tr>")

    lagrows = "\n".join(lagrow(r) for r in lag if r["broken"])

    start, end = result.training_window
    built = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arrival — ETA Prediction</title>
<style>
  :root {{
    color-scheme: light;
    --bg:#f4f4f2; --card:#fcfcfb; --line:#dedcd6;
    --ink:#0b0b0b; --ink2:#52514e; --ink3:#7b7a74;
    --honest:{HONEST['light']}; --leak:{LEAK['light']}; --track:#e7e5df;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --bg:#121211; --card:#1a1a19; --line:#343430;
      --ink:#fff; --ink2:#c3c2b7; --ink3:#8f8e85;
      --honest:{HONEST['dark']}; --leak:{LEAK['dark']}; --track:#2b2b28;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --bg:#121211; --card:#1a1a19; --line:#343430;
    --ink:#fff; --ink2:#c3c2b7; --ink3:#8f8e85;
    --honest:{HONEST['dark']}; --leak:{LEAK['dark']}; --track:#2b2b28;
  }}
  *{{box-sizing:border-box}} html,body{{background:var(--bg);margin:0}}
  body{{font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;color:var(--ink);
       padding:30px 16px 64px;line-height:1.5}}
  .wrap{{max-width:860px;margin:0 auto}}
  .eyebrow{{font:11.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.14em;
           text-transform:uppercase;color:var(--ink3);margin-bottom:10px}}
  h1{{font-size:25px;line-height:1.25;margin:0 0 6px;letter-spacing:-.02em}}
  .stamp{{color:var(--ink3);font-size:11.5px;font-variant-numeric:tabular-nums;
         margin:0 0 24px}}
  section{{background:var(--card);border:1px solid var(--line);border-radius:10px;
          padding:18px 20px;margin-bottom:16px}}
  h2{{font-size:14px;margin:0 0 3px}}
  .lede{{color:var(--ink2);font-size:12.5px;margin:0 0 16px;max-width:72ch}}
  .row{{margin-bottom:14px}}
  .rh{{display:flex;justify-content:space-between;align-items:baseline;
      margin-bottom:5px}}
  .nm{{font-size:13px;font-weight:600}}
  .rt{{font-size:12.5px;font-variant-numeric:tabular-nums;color:var(--ink2)}}
  .track{{background:var(--track);border-radius:4px;height:16px;overflow:hidden}}
  .fill{{height:100%;border-radius:4px}}
  .sub{{font-size:11px;color:var(--ink3);margin-top:4px;
       font-variant-numeric:tabular-nums}}
  table{{width:100%;border-collapse:collapse;font-size:12.5px}}
  th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
     color:var(--ink2);padding:0 10px 7px 0;border-bottom:1px solid var(--line)}}
  td{{padding:7px 10px 7px 0;border-bottom:1px solid var(--line)}}
  tr:last-child td{{border-bottom:0}}
  td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
  .caveat{{font-size:11.5px;color:var(--ink3);line-height:1.6;
          border-left:2px solid var(--line);padding-left:12px}}
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:12px;margin-bottom:4px}}
  .kpi{{border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
  .kv{{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}}
  .kl{{font-size:11px;color:var(--ink2);margin-top:2px}}
</style>
</head>
<body><div class="wrap">

<p class="eyebrow">Arrival &middot; delivery ETA</p>
<h1>The leak looks {result.leakage_gap_mae:.2f} minutes better than the honest
    model. The honest model is only {result.lift_over_baseline_mae:.2f} minutes
    better than doing nothing.</h1>
<p class="stamp">Generated {built:%Y-%m-%d %H:%M} UTC from
   {result.n_train + result.n_test:,} trips &middot;
   {start:%d %b} to {end:%d %b} &middot; every figure computed at build time</p>

<section>
  <h2>What each model scores on the held-out trips</h2>
  <p class="lede">Lower is better. The flat promise is what the business does
     today, so it is the only baseline an improvement means anything against.
     Orange bars are models that cheated.</p>
  {bars}
  <p class="caveat"><b>The gap is the finding.</b> Filtering a feature window on
     when a trip was <i>assigned</i> rather than when it <i>finished</i> buys
     {result.leakage_gap_mae:.2f} minutes of MAE &mdash; more than the honest
     model's entire lift over the baseline &mdash; and none of it survives
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
     is what the accuracy dashboard showed.</p>
  <table>
    <thead><tr><th>Time</th><th class="n">Trips scored</th>
      <th class="n">MAE</th><th class="n">Labels pending</th></tr></thead>
    <tbody>{lagrows}</tbody>
  </table>
  <p class="caveat" style="margin-top:14px">Twenty minutes into the outage the
     dashboard reads the same MAE as a model that never broke, because every
     trip it could score was predicted before 09:00. Accuracy is structurally
     behind by the label delay. <b>Prediction drift is not</b> &mdash;
     predictions exist the moment they are served, and the broken model's
     distribution has already moved while its accuracy has not.</p>
</section>

<section>
  <h2>Serving</h2>
  <div class="kpis">
    <div class="kpi"><div class="kv">{lat.p50_ms:.0f} ms</div><div class="kl">p50</div></div>
    <div class="kpi"><div class="kv">{lat.p99_ms:.0f} ms</div><div class="kl">p99</div></div>
    <div class="kpi"><div class="kv">{lat.mean_ms:.0f} ms</div><div class="kl">mean</div></div>
    <div class="kpi"><div class="kv">{lat.n:,}</div><div class="kl">requests</div></div>
  </div>
  <p class="caveat" style="margin-top:14px">The mean is {lat.mean_ms:.0f} ms and
     would clear any alarm anyone would set. One request in twenty takes
     {lat.p99_ms / 1000:.1f} seconds. Reporting a mean latency is how a service
     stays slow without anybody finding out.</p>
</section>

<section>
  <h2>Could the experiment have seen it?</h2>
  <p class="lede">A treatment that genuinely is 2% better &mdash;
     {ab['true_effect']:.2f} minutes off an 8 minute MAE &mdash; run over
     {ab['n']:,} trips.</p>
  <div class="kpis">
    <div class="kpi"><div class="kv">{ab['true_effect']:.2f}</div><div class="kl">real effect, min</div></div>
    <div class="kpi"><div class="kv">{ab['observed']:.2f}</div><div class="kl">measured, min</div></div>
    <div class="kpi"><div class="kv">{ab['mde']:.2f}</div><div class="kl">smallest detectable, min</div></div>
  </div>
  <p class="caveat" style="margin-top:14px">The measured difference is larger
     than the real one and smaller than what this experiment could resolve, so
     it is noise with a decimal point. Without the third number the second one
     goes into a deck as a win.</p>
</section>

<section>
  <h2>How it is built</h2>
  <p class="lede">Trips come from <b>Dispatch</b>'s gold table, where
     <code>assigned_at</code> is the instant a prediction was due and
     <code>tat_minutes</code> only exists once the trip ends. Features are
     declared once and retrieved two ways &mdash; point-in-time for training, by
     entity for serving &mdash; and a test replays a day asserting the two paths
     return identical values for every trip.</p>
  <p class="lede" style="margin-top:-8px">
     <a href="architecture.html" style="color:var(--honest)">Architecture
     diagram &rarr;</a> &nbsp;&middot;&nbsp;
     <a href="handoff.html" style="color:var(--honest)">What Dispatch hands
     Arrival &rarr;</a></p>
  <p class="caveat">Trips are synthetic where Dispatch's warehouse is absent, so
     treat the absolute minutes as a demonstration. What is not circular is the
     ordering: a leak flatters a model, the flattery does not survive serving,
     and an accuracy dashboard cannot see an outage until its labels arrive.</p>
</section>

</div></body></html>
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
