# Spec: faro per-cell optogenetic inference server

## Purpose

A standalone HTTP service that hosts a loaded predictive model. On each acquired frame of a
field-of-view (FOV), faro sends the **latest per-cell features** for that FOV. The server returns a
**per-cell stimulation exposure** — a light dose expressed as **time in milliseconds** — which
faro uses to drive the next optogenetic stimulation (each cell is illuminated for its own predicted
duration via a DMD).

The server owns the model and any temporal state. faro is a thin client: it extracts features,
calls the server, and applies the returned exposures. This document is the **contract**; faro is
being built to it in parallel.

## Role in the loop

```
faro:   acquire frame ─► segment ─► track (assigns particle IDs) ─► extract features
                                                                        │
                                                                        ▼  POST /predict
server: receive per-cell features ─► update per-cell temporal state ─► model ─► per-cell exposure
                                                                        │
                                                                        ▼  JSON response
faro:   build DMD staircase ─► stimulate each cell for its exposure ─► next frame …
```

## Transport

- **Protocol:** HTTP/1.1, JSON request and response (a `parquet`/binary variant may be added later;
  start with JSON).
- **Runs on:** the compute cluster (GPU node). faro connects over the network, commonly via an SSH
  tunnel to a cluster node. The server should bind a configurable host/port (e.g. `0.0.0.0:8080`).
- **Suggested stack:** FastAPI + uvicorn (async, easy concurrency, auto request validation). Any
  equivalent is fine as long as the endpoints/schemas below match.

## Endpoints

### `POST /predict`  (the main call)
Input: per-cell features for **one FOV at one timepoint**. Output: exposure per cell.

**Request body:**
```json
{
  "fov": 3,
  "timestep": 17,
  "time": 1020.0,
  "cells": [
    {"particle": 42, "x": 512.3, "y": 128.9, "cnr": 1.83, "area_nuc": 210.0,
     "mean_intensity_C1_nuc": 812.0, "mean_intensity_C1_ring": 1490.0},
    {"particle": 43, "x": 640.1, "y": 300.2, "cnr": 0.94, "area_nuc": 188.0,
     "mean_intensity_C1_nuc": 733.0, "mean_intensity_C1_ring": 690.0}
  ]
}
```
- `fov` (int): field-of-view index. **State must be namespaced by `fov`.**
- `timestep` (int): monotonically increasing frame index within the experiment for this FOV.
- `time` (float, optional): acquisition wall-clock/experiment time in seconds (for models that
  need real dt between frames).
- `cells` (list): one object per segmented+tracked cell. Always contains `particle` (int, the
  **stable per-FOV cell identity**) plus a set of feature columns. **The exact feature set is not
  fixed** — treat it as a dict and select the columns your model needs. Typical ERK-KTR columns:
  `x`, `y`, `area_nuc`, `cnr`, `cnr_median`, `mean_intensity_C0_nuc`, `mean_intensity_C1_nuc`,
  `median_intensity_C0_nuc`, `mean_intensity_C0_ring`, `mean_intensity_C1_ring`,
  `ref_mean_intensity`. Design the model I/O to be robust to missing/extra columns.

**Response body:**
```json
{
  "fov": 3,
  "timestep": 17,
  "exposures": {"42": 250.0, "43": 0.0}
}
```
- `exposures` (object): maps **`particle` (as string key)** → **exposure in milliseconds** (float).
  - Valid range: **`0` to ~`3000` ms**. `0` (or omission) = **do not stimulate that cell**.
  - Values below faro's DMD-switch granularity (~25 ms) may be quantized by faro; returning fine
    values is fine.
  - A cell present in the request but **absent** from `exposures` is treated by faro as `0` (no
    stim). Prefer explicit `0.0` for clarity.
  - Do **not** invent particles that weren't in the request.

### `GET /health`
Returns `200 {"status": "ok", "model_loaded": true}` when the model is loaded and ready. faro (and
ops) use this for readiness.

### `POST /reset`  (recommended)
Clears all per-cell temporal state (e.g. `{"fov": 3}` to reset one FOV, or empty body to reset
all). Called by faro at the **start of a new experiment/run** so stale history from a prior run
doesn't leak in. Returns `200`.

### `GET /info`  (optional)
Returns model metadata: model name/version, expected feature columns, output units. Useful for
logging and reproducibility.

## Statefulness — the important part

faro sends only the **latest frame's** features, not history. If the model needs temporal context
(previous exposures, feature trajectories, time since last stim), the **server must accumulate it**,
keyed on **`(fov, particle)`**.

Requirements:
- **Namespace all state by `fov`.** `particle` IDs are only unique within a FOV; the same integer
  in FOV 1 and FOV 2 are different cells.
- **Tolerate identity churn.** Tracking is imperfect: a `particle` can appear (birth / new cell or
  a re-linked track), disappear (death / lost track), or — rarely — a cell's identity may break and
  reappear under a new `particle`. The model must handle first-time-seen particles (no history) and
  not crash on gaps.
- **Optional lineage:** faro's tracker may add a `parent_particle` column on cell divisions; if
  present in `cells`, the server may use it to seed a daughter's state from its mother. Treat as
  optional.
- Consider bounding memory (e.g. evict particles not seen for N frames).

## Concurrency, ordering, latency

- **Concurrency:** faro runs its pipeline in a thread pool and may call `/predict` for **up to ~4
  FOVs concurrently**. The server must handle concurrent requests safely. Since different FOVs have
  independent state, per-FOV work is independent; if the GPU model isn't thread-safe, serialize
  inference behind a lock/queue while keeping the HTTP layer concurrent.
- **Ordering:** for a given FOV, calls arrive in increasing `timestep` order under normal operation,
  but do not hard-assume it — guard against out-of-order or duplicate `timestep`s (see retries).
- **Idempotency / retries:** faro retries a failed call up to **5 times with ~0.5 s backoff**. A
  retried call repeats the same `(fov, timestep)`. Make state updates **idempotent per
  `(fov, timestep)`** (e.g. record the last processed timestep per FOV and skip/replace rather than
  double-appending) so retries don't corrupt history.
- **Latency budget:** faro's acquisition interval is **60 s** and it blocks up to 80 s for a
  response, so latency is not tight — but aim for well under a few seconds per call. Cold-start
  (model load) should happen at startup, not on first request; `/health` should report not-ready
  until loaded.

## faro-side contract (for reference)

faro calls the server through a swappable `InferenceClient` interface. The HTTP implementation maps
directly to the schema above:

```python
class InferenceClient(Protocol):
    def predict(self, cells: pd.DataFrame, meta: dict) -> dict[int, float]:
        """cells: one row per cell (incl. `particle` + feature columns).
        meta: {"fov": int, "timestep": int, "time": float, ...}.
        Returns {particle_id: exposure_ms}. Missing particles => 0 (no stim)."""
```

`HttpInferenceClient` serializes `cells` + `meta` into the `/predict` request, applies the 5×/0.5 s
retry loop, and parses `exposures` back into `{int(particle): float(ms)}`. A `FakeInferenceClient`
(in-process, e.g. `exposure = f(cnr)`) is used for faro's tests and for development before the real
server exists — so the server team can validate independently and faro can integrate against the
fake meanwhile.

## Deliverables for the server agent

1. FastAPI (or equivalent) app implementing `/predict`, `/health`, `/reset`, optional `/info`,
   matching the schemas above exactly.
2. A pluggable model-loading seam: load the trained model once at startup; a clear function
   `predict_exposures(fov, timestep, cells, state) -> {particle: ms}` where the real model plugs in.
   Ship a trivial stub model first (e.g. exposure as a monotonic function of `cnr`, clamped to
   [0, 3000] ms) so the end-to-end path works before the real model is trained.
3. Per-`(fov, particle)` temporal state store with idempotent `(fov, timestep)` updates and reset.
4. Concurrency-safe inference (HTTP concurrent; serialize GPU if needed).
5. A small client-side smoke test / example request (curl or Python) reproducing the example
   payloads above.
6. Deployment notes: how to launch on the cluster GPU node, the bind host/port, and the SSH-tunnel
   command faro will use to reach it.

## Explicitly out of scope for the server

- Image acquisition, segmentation, tracking, DMD control — all faro's side.
- Building the DMD "staircase" from the exposures — faro does that; the server only returns
  `{particle: exposure_ms}`.
- Training the predictive model (separate effort); the server just hosts the loaded model.
