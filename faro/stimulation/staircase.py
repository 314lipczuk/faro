"""Cumulative "staircase" decomposition of per-cell exposures into DMD sub-frames.

A DMD pattern is binary and the stim dose is the LED-on time, shared by every
lit cell. To deliver a *different* exposure to each cell we split one stim
timepoint into a sequence of sub-frames and let each cell drop out of the
pattern once it has reached its target dose:

    all cells ON ──(e_1)──► cells with exp ≥ e_2 ON ──(e_2-e_1)──► …

Sub-frame ``i`` shows the mask of every cell whose target exposure is ``≥ e_i``
and is displayed for ``e_i - e_{i-1}``. A cell therefore accumulates on-time
equal to its target exposure, and the **total** wall-clock is the *maximum*
exposure (not the sum) — independent of cell count or number of levels.

``build_staircase`` is pure and array-only so it can be unit-tested without a
microscope. The stimulator is responsible for turning per-cell (``particle``)
exposures into the per-``label`` mapping this module consumes.
"""

from __future__ import annotations

import numpy as np


def build_staircase(
    labels: np.ndarray,
    label_exposures: dict[int, float],
    *,
    eps_ms: float = 25.0,
    max_subframes: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Decompose per-cell exposures into cumulative ``(mask, duration_ms)`` sub-frames.

    Args:
        labels: current-frame segmentation label image (HxW); pixel value is a
            per-frame ``label`` id (0 = background).
        label_exposures: maps ``label`` id -> target exposure in ms. Labels not
            present, mapped to <= 0, or NaN are not stimulated.
        eps_ms: exposures are snapped to a grid of this size, merging levels that
            differ by less than ``eps_ms`` to bound the number of DMD switches.
            A positive exposure never snaps to 0.
        max_subframes: optional hard cap on the number of sub-frames; if the
            grid yields more levels, they are collapsed (by quantile) and each
            cell snapped to the nearest surviving level.

    Returns:
        List of ``(mask, duration_ms)`` in increasing-threshold order, where
        ``mask`` is a uint8 HxW array (1 = illuminate). Empty if no cell has a
        positive exposure.
    """
    labels = np.asarray(labels)
    if labels.size == 0 or not label_exposures:
        return []

    lab_ids = np.array(list(label_exposures.keys()))
    exposures = np.array(list(label_exposures.values()), dtype=float)

    positive = np.isfinite(exposures) & (exposures > 0)
    if not positive.any():
        return []

    snapped = np.zeros_like(exposures)
    snapped[positive] = _snap_exposures(exposures[positive], eps_ms, max_subframes)

    thresholds = np.unique(snapped[snapped > 0])
    if thresholds.size == 0:
        return []
    thresholds.sort()

    # Per-pixel exposure image via a label -> exposure lookup table.
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=float)
    for lab, exp in zip(lab_ids, snapped):
        lab = int(lab)
        if 0 <= lab <= max_label:
            lut[lab] = exp
    exposure_image = lut[labels]

    subframes: list[tuple[np.ndarray, float]] = []
    prev = 0.0
    for thr in thresholds:
        mask = (exposure_image >= thr - 1e-9).astype(np.uint8)
        subframes.append((mask, float(thr - prev)))
        prev = float(thr)
    return subframes


def _snap_exposures(
    exposures: np.ndarray, eps_ms: float, max_subframes: int | None
) -> np.ndarray:
    """Snap positive exposures onto a coarse grid, optionally capping level count."""
    exposures = np.asarray(exposures, dtype=float)
    if eps_ms and eps_ms > 0:
        snapped = np.round(exposures / eps_ms) * eps_ms
        # A requested stim must never be silently dropped by snapping to 0.
        snapped = np.where(snapped <= 0, float(eps_ms), snapped)
    else:
        snapped = exposures.copy()

    levels = np.unique(snapped)
    if max_subframes is not None and levels.size > max_subframes:
        # Collapse to <= max_subframes representative levels (quantiles), then
        # snap each cell to the nearest surviving level.
        qs = np.quantile(snapped, np.linspace(0.0, 1.0, max_subframes + 1)[1:])
        levels = np.unique(qs)
        idx = np.abs(snapped[:, None] - levels[None, :]).argmin(axis=1)
        snapped = levels[idx]
    return snapped
