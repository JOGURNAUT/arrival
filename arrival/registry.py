"""Saved models, named after what is in them.

A version here is a hash of the model's content -- its kind, its weights, its
feature order, the window it was fitted on. Not a counter, and not a name
somebody typed. Two reasons, both learned the hard way:

  * `v3` tells you nothing about which model is in production. A content hash
    means the artifact on the server, the row in the metrics table and the
    model in the notebook can be compared by eye and shown to be the same
    thing, or shown not to be.
  * A counter can be reused. Overwriting `v3` with a different model leaves
    every prediction already logged against `v3` attributed to weights that no
    longer exist, and nothing in the logs says so. Here the same content is
    always the same version, and different content cannot take a version that
    is already occupied.

`created_at` and the metrics sit outside the hash deliberately. Retraining the
same data to the same weights should produce the same version, and it would not
if the clock were part of the name.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

from .contracts import ArrivalError, Metrics

ARTIFACT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
CARD_NAME = "model.json"
VERSION_LENGTH = 12


def version_of(payload: dict[str, Any],
               training_window: tuple[dt.datetime, dt.datetime]) -> str:
    """The version a given model and training window must be called."""
    content = json.dumps(
        {"payload": payload,
         "training_window": [training_window[0].isoformat(),
                             training_window[1].isoformat()]},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:VERSION_LENGTH]


def save(model, metrics: Metrics,
         training_window: tuple[dt.datetime, dt.datetime],
         root: pathlib.Path | str | None = None) -> str:
    """Write a model to `artifacts/<version>/model.json` and return the version.

    Saving the same model twice is a no-op rather than an error, because a
    retrain that changed nothing should not need special handling anywhere. A
    *different* model under an existing version is refused: that can only
    happen if the hash stopped covering something it needs to cover, and
    silently overwriting would be the worst available response.
    """
    payload = model.payload()
    version = version_of(payload, training_window)
    directory = pathlib.Path(root or ARTIFACT_ROOT) / version
    card_path = directory / CARD_NAME

    card = {
        "version": version,
        "payload": payload,
        "feature_names": list(payload.get("feature_names", [])),
        "training_window": [training_window[0].isoformat(),
                            training_window[1].isoformat()],
        "metrics": metrics.as_row(),
        # To the microsecond, because two models saved in the same second
        # would otherwise tie, and `latest` would resolve the tie by hash --
        # which means deploying whichever of them sorts first.
        "created_at": dt.datetime.now().isoformat(timespec="microseconds"),
    }

    if card_path.exists():
        existing = json.loads(card_path.read_text(encoding="utf-8"))
        if existing.get("payload") != payload:
            raise ArrivalError(
                f"{version} already holds a different model; the version hash "
                f"is not covering everything that distinguishes them")
        return version

    directory.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    model.version = version
    return version


def read_card(version: str,
              root: pathlib.Path | str | None = None) -> dict[str, Any]:
    card_path = pathlib.Path(root or ARTIFACT_ROOT) / version / CARD_NAME
    if not card_path.exists():
        raise ArrivalError(f"no artifact {version} under "
                           f"{pathlib.Path(root or ARTIFACT_ROOT)}")
    return json.loads(card_path.read_text(encoding="utf-8"))


def load(version: str, root: pathlib.Path | str | None = None):
    """Rebuild a saved model, feature order included.

    The feature names come back in the order they were trained in, because
    they were stored as a list and a list is ordered. A model served columns in
    a different order than it was fitted on does not fail -- it answers, and
    the answers are wrong by an amount nobody can see. `FeatureVector.ordered`
    is what enforces this at prediction time; this function's job is to make
    sure it has the right list to enforce.
    """
    card = read_card(version, root)
    payload = card["payload"]
    kinds = _model_kinds()
    kind = payload.get("kind")
    if kind not in kinds:
        raise ArrivalError(f"artifact {version} is a {kind!r}, which this "
                           f"registry cannot rebuild")
    return kinds[kind].from_payload(payload, card["version"])


def versions(root: pathlib.Path | str | None = None) -> list[str]:
    """Every saved version, oldest first by when it was written."""
    base = pathlib.Path(root or ARTIFACT_ROOT)
    if not base.exists():
        return []
    found = []
    for directory in base.iterdir():
        card_path = directory / CARD_NAME
        if not card_path.is_file():
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        found.append((card.get("created_at", ""), card.get("version",
                                                           directory.name)))
    return [version for _created, version in sorted(found)]


def latest(root: pathlib.Path | str | None = None) -> str:
    """The most recently saved version.

    Resolved by `created_at` rather than by filesystem mtime: copying the
    artifacts directory to another machine rewrites every mtime and would
    reorder the models, which is a confusing way to deploy the wrong one.
    """
    found = versions(root)
    if not found:
        raise ArrivalError(f"no artifacts under "
                           f"{pathlib.Path(root or ARTIFACT_ROOT)}")
    return found[-1]


def _model_kinds() -> dict[str, Any]:
    """Kind name to class.

    Imported inside the function because the trainer imports the registry to
    save what it fitted, and a module-level import back into the trainer would
    close that loop. The registry is the lower layer and stays importable on
    its own.
    """
    from .train import BaselineModel, RidgeModel
    return {"baseline": BaselineModel, "ridge": RidgeModel}
