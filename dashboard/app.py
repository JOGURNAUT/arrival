"""The Arrival dashboard: train, serve and watch a model go wrong.

    streamlit run dashboard/app.py

This is the live counterpart to docs/index.html. The static page is what belongs
in a link because it outlives everything; this is what belongs in a screen share,
because you can move the break time and watch the blind window move with it.

Nothing here recomputes a metric. Every number comes from `arrival.train`,
`arrival.monitor` or `arrival.serve` -- the same code the tests exercise. A
dashboard that derives its own figures is a second definition of every metric it
shows, and the two drift.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import random
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrival import source, train                        # noqa: E402
from arrival.contracts import Prediction, Trip            # noqa: E402
from arrival.experiment import minimum_detectable_effect  # noqa: E402
from arrival.monitor import Monitor                       # noqa: E402

st.set_page_config(page_title="Arrival", page_icon="🛵", layout="wide")


@st.cache_data(ttl=600)
def trained():
    trips = source.load()
    return train.run(trips), len(trips)


st.title("Arrival")
st.caption("ETA prediction with a feature store, on Dispatch's trip telemetry. "
           "Every figure is read from the library — nothing on this page is "
           "recomputed in the dashboard.")

result, n_trips = trained()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trips", f"{n_trips:,}")
c2.metric("Flat promise MAE", f"{result.metrics['baseline'].mae:.2f} min")
c3.metric("Model MAE", f"{result.metrics['ridge'].mae:.2f} min",
          delta=f"-{result.lift_over_baseline_mae:.2f} min", delta_color="inverse")
c4.metric("Leak looks better by", f"{result.leakage_gap_mae:.2f} min",
          help="None of it survives production.")

models_tab, outage_tab, ab_tab = st.tabs(
    ["Models and the leak", "The blind window", "Experiment"])

with models_tab:
    st.caption("Lower is better. The flat promise is what the business does "
               "today, so it is the only baseline an improvement means "
               "anything against.")
    rows = [("flat promise", result.metrics["baseline"]),
            ("ridge, point-in-time", result.metrics["ridge"]),
            ("ridge, leaky: in-flight", result.metrics["ridge_leaky_in_flight"]),
            ("ridge, leaky: no window", result.metrics["ridge_leaky_lifetime"])]
    df = pd.DataFrame([{"model": n, **m.as_row()} for n, m in rows])
    st.bar_chart(df.set_index("model")["mae"], height=300)
    st.dataframe(df, width="stretch", hide_index=True)
    st.info(
        f"The honest model takes **{result.lift_over_baseline_mae:.3f} min** off "
        f"the flat promise. Filtering the feature window on `assigned_at` "
        f"instead of `delivered_at` looks **{result.leakage_gap_mae:.3f} min** "
        f"better again — more than the entire real improvement — and buys "
        f"nothing, because one of the trips it reads is the trip being "
        f"predicted. Dropping the window altogether gaps only "
        f"**{result.lifetime_leakage_gap_mae:.3f} min**: small enough to "
        f"dismiss as noise, wrong for the same reason, and the shape a GROUP BY "
        f"without a date filter actually takes.")

with outage_tab:
    st.caption("A model starts doubling every estimate. Move the controls and "
               "watch how long the accuracy dashboard stays green.")
    col_a, col_b = st.columns(2)
    trip_minutes = col_a.slider("How long a trip takes (min)", 10, 60, 35)
    elapsed = col_b.slider("Minutes since the model broke", 5, 120, 20, step=5)

    break_at = dt.datetime(2026, 9, 1, 9, 0)
    scored = []
    for broken in (False, True):
        mon = Monitor()
        trips = []
        for i in range(260):
            at = dt.datetime(2026, 9, 1, 8, 0) + dt.timedelta(minutes=i)
            t = Trip(trip_id=f"T{i:04d}", store_id="NORTHGATE",
                     driver_id=f"D{i % 20:03d}", assigned_at=at,
                     distance_m=3000, promised_minutes=19,
                     delivered_at=at + dt.timedelta(minutes=trip_minutes),
                     tat_minutes=float(trip_minutes))
            guess = trip_minutes * 2 if (broken and at >= break_at) \
                else trip_minutes + 0.5
            mon.record_prediction(
                Prediction(trip_id=t.trip_id, minutes=guess,
                           model_version="v1", served_at=at), t)
            trips.append(t)
        for t in trips:
            mon.record_outcome(t)
        as_of = break_at + dt.timedelta(minutes=elapsed)
        m = mon.realised_accuracy(as_of)
        scored.append({"model": "broken" if broken else "healthy",
                       "trips scored": m.n if m else 0,
                       "MAE": round(m.mae, 3) if m else None,
                       "labels pending": mon.labels_pending(as_of)})

    st.dataframe(pd.DataFrame(scored), width="stretch", hide_index=True)
    healthy, broken_row = scored[0]["MAE"], scored[1]["MAE"]
    if healthy == broken_row:
        st.error(f"**{elapsed} minutes into the outage the broken model and the "
                 f"healthy model read the same MAE.** Every trip scored so far "
                 f"was predicted before the break; their labels are the only "
                 f"ones that have arrived.")
    else:
        st.success(f"After {elapsed} minutes the break is visible: "
                   f"{broken_row} against {healthy}.")
    st.caption("Accuracy is structurally behind by the label delay. Prediction "
               "drift is not — predictions exist the moment they are served, "
               "which is why it is the signal that catches this first.")

with ab_tab:
    st.caption("A treatment that genuinely is better. The question is whether "
               "the experiment could have told you.")
    col_a, col_b = st.columns(2)
    per_arm = col_a.slider("Trips per arm", 100, 20_000, 400, step=100)
    improvement = col_b.slider("True improvement (%)", 1, 25, 2)

    rng = random.Random(7)
    base, std = 8.0, 5.55
    control = [max(0.5, rng.gauss(base, std)) for _ in range(per_arm)]
    treatment = [max(0.5, rng.gauss(base * (1 - improvement / 100), std))
                 for _ in range(per_arm)]
    observed = sum(control) / per_arm - sum(treatment) / per_arm
    detectable = minimum_detectable_effect(std, per_arm, per_arm)
    real = base * improvement / 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Real effect", f"{real:.2f} min")
    m2.metric("Measured", f"{observed:.2f} min")
    m3.metric("Smallest detectable", f"{detectable:.2f} min")

    if abs(observed) < detectable:
        st.warning("**Inconclusive.** The measured difference is inside what "
                   "this experiment could resolve, so it is consistent with "
                   "there being no difference at all — however neat the point "
                   "estimate looks.")
    else:
        st.success("The measured difference exceeds the minimum detectable "
                   "effect, so it is worth reporting.")
    st.caption("Resolution improves with the square root of the sample, which "
               "is why quadrupling traffic only halves the detectable effect.")

st.divider()
st.caption("Trips come from Dispatch's warehouse where present, a seeded "
           "generator otherwise — so treat the absolute minutes as a "
           "demonstration. What is not circular is the ordering.")
