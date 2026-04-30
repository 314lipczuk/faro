"""Per-cell random patch stimulation.

For every segmented cell the FOV is tiled into a regular grid of
``PATCH_SIZE``×``PATCH_SIZE`` px patches (default 14 px). Patches that are
not entirely inside the cell's segmentation mask are discarded. One of the
remaining patches is picked at random per cell and stamped with a
``DOT_DIAMETER`` px (default 10 px) circular dot in the stim mask.

The patch selection is taken **once per FOV** — at the first stim frame —
and the same fixed-in-image-space mask is then reused for every subsequent
stim frame in that FOV. This lets you stimulate the same spots across a
contiguous block of stim frames (e.g. 10 in a row) without the position
hopping each frame. The seed is ``(fov,)``-only so a replay of the same
FOV gives the same patches.

The per-cell patch coordinates (cell label, patch grid index, patch bbox,
dot center) are accumulated in ``self.records`` on **every** call, so the
notebook's merge on ``(fov, fov_timestep, label)`` attaches patch info to
each stim frame's row in ``exp_data.parquet`` — no separate sidecar file
is written.
"""

import threading

import numpy as np
import pandas as pd

from .base import StimWithPipeline


class RandomStimPerCell14pxPatches(StimWithPipeline):
    PATCH_SIZE = 14
    DOT_DIAMETER = 10

    def __init__(self, *, seed: int = 0):
        if self.DOT_DIAMETER > self.PATCH_SIZE:
            raise ValueError("DOT_DIAMETER must be <= PATCH_SIZE")
        self._base_seed = int(seed)
        self.records: list[dict] = []
        self._lock = threading.Lock()
        # Per-FOV cache of {cell_label: (patch_i, patch_j)}, populated on
        # the first stim frame for that FOV and reused on every subsequent
        # stim frame. Keeps the illuminated spots fixed across the stim
        # block instead of rerolling each frame.
        self._fov_selections: dict[int, dict[int, tuple[int, int]]] = {}

        # Pre-compute the dot footprint inside a PATCH_SIZE x PATCH_SIZE tile.
        center = (self.PATCH_SIZE - 1) / 2
        radius = self.DOT_DIAMETER / 2
        yy, xx = np.ogrid[: self.PATCH_SIZE, : self.PATCH_SIZE]
        self._dot = ((yy - center) ** 2 + (xx - center) ** 2) <= radius**2

    def _rng_for(self, metadata):
        # Per-FOV deterministic seed (no timestep dependence) so the
        # selection is stable across consecutive stim frames within an FOV.
        fov = int(metadata.get("fov", 0)) if metadata else 0
        seed = (self._base_seed * 2_654_435_761 + fov * 1_000_003) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _select_patch_per_cell(self, labels, rng):
        ps = self.PATCH_SIZE
        h, w = labels.shape
        n_h, n_w = h // ps, w // ps
        if n_h == 0 or n_w == 0:
            return {}

        cropped = labels[: n_h * ps, : n_w * ps]
        blocks = cropped.reshape(n_h, ps, n_w, ps)
        block_min = blocks.min(axis=(1, 3))
        block_max = blocks.max(axis=(1, 3))
        # A patch is fully inside one cell iff all its pixels share the same
        # non-background label.
        uniform = (block_min == block_max) & (block_min != 0)
        if not uniform.any():
            return {}

        ii, jj = np.nonzero(uniform)
        cell_labels = block_min[ii, jj]

        per_cell: dict[int, tuple[int, int]] = {}
        for label in np.unique(cell_labels):
            mask = cell_labels == label
            choices = np.flatnonzero(mask)
            chosen = int(rng.integers(choices.size))
            per_cell[int(label)] = (int(ii[choices[chosen]]), int(jj[choices[chosen]]))
        return per_cell

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        ps = self.PATCH_SIZE
        meta = metadata or {}
        fov = meta.get("fov")
        fov_timestep = meta.get("fov_timestep", meta.get("timestep"))

        # First stim frame for this FOV picks the patches; subsequent
        # stim frames within the same FOV reuse them so the illuminated
        # spots stay fixed in image space.
        cache_key = int(fov) if fov is not None else 0
        if cache_key not in self._fov_selections:
            rng = self._rng_for(meta)
            self._fov_selections[cache_key] = self._select_patch_per_cell(labels, rng)
        per_cell = self._fov_selections[cache_key]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)

        rows = []
        for cell_label, (i, j) in per_cell.items():
            y0, x0 = i * ps, j * ps
            y1, x1 = y0 + ps, x0 + ps
            tile = stim_mask[y0:y1, x0:x1]
            np.maximum(tile, self._dot.astype(np.uint8) * 255, out=tile)
            rows.append(
                {
                    "fov": fov,
                    "fov_timestep": fov_timestep,
                    "label": cell_label,
                    "patch_i": i,
                    "patch_j": j,
                    "patch_y_min": y0,
                    "patch_x_min": x0,
                    "patch_y_max": y1,
                    "patch_x_max": x1,
                    "patch_dot_y": y0 + (ps - 1) / 2,
                    "patch_dot_x": x0 + (ps - 1) / 2,
                }
            )

        if rows:
            with self._lock:
                self.records.extend(rows)

        return stim_mask, None

    def to_dataframe(self) -> pd.DataFrame:
        """Per-cell patch selections accumulated during the run.

        Merge into ``exp_data.parquet`` on ``(fov, fov_timestep, label)`` to
        attach the selected patch bbox to each cell's stim-frame row.
        """
        with self._lock:
            return pd.DataFrame(list(self.records))
