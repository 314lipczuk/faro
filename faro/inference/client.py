"""Inference-server clients.

faro sends the latest per-cell features for a FOV to an inference server that
hosts a loaded predictive model, and receives a per-cell **stimulation exposure**
(a light dose expressed as time in milliseconds). The transport is abstracted
behind the :class:`InferenceClient` interface so the same stimulator works with
an in-process fake, an HTTP server, or (later) a ZMQ endpoint.

All clients return ``dict[particle_id -> exposure_ms]``. A particle omitted from
the result is treated as ``0`` ms (no stimulation) by the caller.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# faro's DMD dose is delivered as LED time; keep exposures within a sane band.
# The upper bound matches the Niesen rig's practical stim range (25 ms - 3 s).
DEFAULT_EXPOSURE_CLIP = (0.0, 3000.0)


@runtime_checkable
class InferenceClient(Protocol):
    """Maps this frame's per-cell features to a per-cell stim exposure (ms).

    Implementations must be safe to call concurrently: faro runs the pipeline on
    a thread pool, so ``predict`` may be invoked for several FOVs at once.
    """

    def predict(self, cells: pd.DataFrame, meta: dict) -> dict[int, float]:
        """Return ``{particle_id: exposure_ms}`` for the cells in *cells*.

        Args:
            cells: one row per cell; must include a ``particle`` column plus
                whatever feature columns the model consumes.
            meta: frame metadata, e.g. ``{"fov", "timestep", "time"}``.

        Returns:
            ``{particle_id: exposure_ms}``. Particles may be omitted; the caller
            treats a missing particle as 0 ms (no stim).
        """
        ...


class FakeInferenceClient:
    """In-process inference client for tests and pre-server development.

    Computes each cell's exposure from a user-supplied ``rule`` callable, so a
    full closed loop can run with no network and no trained model. The rule
    receives one cell's feature row (a ``pandas.Series``) and returns an
    exposure in milliseconds.

    Example::

        # dose proportional to the ERK C/N ratio, clamped to [0, 3000] ms
        client = FakeInferenceClient(rule=lambda c: 500.0 * c.get("cnr", 0.0))
    """

    def __init__(
        self,
        rule: Callable[[pd.Series], float] | None = None,
        *,
        default_exposure_ms: float = 0.0,
        clip: tuple[float, float] = DEFAULT_EXPOSURE_CLIP,
    ):
        self.rule = rule if rule is not None else (lambda _row: default_exposure_ms)
        self.default_exposure_ms = default_exposure_ms
        self.clip = clip

    def predict(self, cells: pd.DataFrame, meta: dict) -> dict[int, float]:
        lo, hi = self.clip
        out: dict[int, float] = {}
        if cells is None or len(cells) == 0 or "particle" not in cells.columns:
            return out
        for _, row in cells.iterrows():
            particle = row["particle"]
            if pd.isna(particle):
                continue
            try:
                exposure = float(self.rule(row))
            except Exception:
                exposure = self.default_exposure_ms
            if math.isnan(exposure):
                exposure = self.default_exposure_ms
            out[int(particle)] = float(min(max(exposure, lo), hi))
        return out


class HttpInferenceClient:
    """POSTs per-cell features to an HTTP inference server; parses doses back.

    Matches the faro/inference-server contract: the request body is
    ``{"fov", "timestep", "time", "cells": [{"particle", ...features}]}`` and the
    response is ``{"exposures": {"<particle>": exposure_ms}}``. Mirrors the retry
    behaviour of :mod:`faro.segmentation.remote` (a few attempts with a short
    backoff) so a transient network blip doesn't drop a stim frame.
    """

    def __init__(
        self,
        server: str,
        *,
        path: str = "/predict",
        timeout: float = 30.0,
        max_attempts: int = 5,
        retry_backoff: float = 0.5,
        clip: tuple[float, float] = DEFAULT_EXPOSURE_CLIP,
    ):
        self.url = server.rstrip("/") + "/" + path.lstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_backoff = retry_backoff
        self.clip = clip

    def predict(self, cells: pd.DataFrame, meta: dict) -> dict[int, float]:
        if cells is None or len(cells) == 0 or "particle" not in cells.columns:
            return {}
        import requests  # local import keeps `requests` optional at module load

        payload = {
            "fov": int(meta.get("fov", 0)),
            "timestep": int(meta.get("timestep", 0)),
            "time": float(meta.get("time", 0.0) or 0.0),
            "cells": _cells_to_records(cells),
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return self._parse(resp.json())
            except Exception as e:  # network error, bad status, or bad JSON
                last_exc = e
                if attempt < self.max_attempts - 1:
                    time.sleep(self.retry_backoff)
        raise RuntimeError(
            f"Inference server {self.url!r} failed after {self.max_attempts} "
            f"attempts"
        ) from last_exc

    def _parse(self, data: dict) -> dict[int, float]:
        lo, hi = self.clip
        exposures = data.get("exposures", {}) if isinstance(data, dict) else {}
        out: dict[int, float] = {}
        for key, value in exposures.items():
            try:
                out[int(key)] = float(min(max(float(value), lo), hi))
            except (TypeError, ValueError):
                continue
        return out


def _json_safe(value):
    """Convert numpy/exotic values to JSON-safe Python; NaN -> None."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.bool_):
        value = bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Unknown/opaque type: stringify rather than crash the whole request.
    return str(value)


def _cells_to_records(cells: pd.DataFrame) -> list[dict]:
    """Serialize a per-cell DataFrame to JSON-safe records."""
    if cells is None or len(cells) == 0:
        return []
    records = cells.to_dict(orient="records")
    return [{k: _json_safe(v) for k, v in rec.items()} for rec in records]
