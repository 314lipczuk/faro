"""Stimulator that asks a deep-learning inference server for per-cell doses.

On each stim frame this stimulator:

1. gathers the current frame's per-cell features (it runs the feature extractor
   itself, since the pipeline merges features *after* the stim-mask compute);
2. sends ``{particle, features...}`` to an :class:`~faro.inference.InferenceClient`
   and gets back ``{particle: exposure_ms}``;
3. turns those per-cell exposures into a cumulative **staircase** of
   ``(mask, duration_ms)`` sub-frames (see :mod:`faro.stimulation.staircase`).

The returned payload is a *list* of ``(mask, duration_ms)`` tuples rather than a
single mask; the controller expands it into one DMD sub-frame per step. Existing
single-mask stimulators are unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StimWithPipeline
from .staircase import build_staircase


class InferenceServerStim(StimWithPipeline):
    """Per-cell optogenetic dosing driven by a remote inference model.

    Args:
        client: an :class:`~faro.inference.InferenceClient` (fake, HTTP, ...).
        feature_extractor: used to compute per-cell features in-hook. If ``None``,
            only whatever columns already exist on ``tracks`` (particle, x, y) are
            sent to the server.
        mask_key: key into ``label_images`` for the segmentation labels.
        eps_ms: staircase quantization grid (merges near-equal exposures).
        max_subframes: optional cap on the number of DMD sub-frames per stim frame.
    """

    def __init__(
        self,
        client,
        feature_extractor=None,
        *,
        mask_key: str = "labels",
        eps_ms: float = 25.0,
        max_subframes: int | None = None,
    ):
        self.client = client
        self.feature_extractor = feature_extractor
        self.mask_key = mask_key
        self.eps_ms = eps_ms
        self.max_subframes = max_subframes

    def get_stim_mask(
        self,
        label_images: dict,
        metadata: dict = None,
        img: np.ndarray = None,
        tracks: "pd.DataFrame | None" = None,
    ) -> tuple[list[tuple[np.ndarray, float]], object]:
        metadata = metadata or {}
        labels = label_images[self.mask_key]

        cells = self._current_cells(label_images, img, tracks, metadata)
        if cells is None or cells.empty or "particle" not in cells.columns:
            return [], None

        meta = {
            "fov": metadata.get("fov"),
            "timestep": metadata.get("timestep"),
            "time": metadata.get("time"),
        }
        exposures = self.client.predict(cells, meta)  # {particle: exposure_ms}
        if not exposures:
            return [], None

        label_exposures = self._label_exposures(cells, exposures)
        subframes = build_staircase(
            labels,
            label_exposures,
            eps_ms=self.eps_ms,
            max_subframes=self.max_subframes,
        )
        return subframes, None

    # Identity/position columns always sent to the server (when present).
    _IDENTITY_COLS = ("particle", "label", "x", "y")

    def _current_cells(self, label_images, img, tracks, metadata) -> "pd.DataFrame | None":
        """Return this frame's per-cell rows: ``particle`` + features only.

        Deliberately slim — identity/position plus the feature columns the
        extractor produced. Broadcast metadata columns on ``tracks``
        (``channels``, ``img_shape``, ...) are dropped so the payload stays
        small and JSON-serializable.
        """
        if tracks is None or tracks.empty:
            return None
        # Restrict to the current frame's cells (tracks accumulates all frames).
        fname = metadata.get("fname")
        if fname is not None and "fname" in tracks.columns:
            current = tracks[tracks["fname"] == fname].copy()
        else:
            current = tracks.copy()
        if current.empty:
            return None

        feature_cols: list[str] = []
        # Merge freshly-computed features (the pipeline hasn't merged them yet).
        if self.feature_extractor is not None and img is not None:
            try:
                features_df, _ = self.feature_extractor.extract_features(
                    label_images, img, tracks, metadata
                )
            except Exception as e:  # a FE crash must not stall stimulation
                print(f"InferenceServerStim: feature extraction failed: {e}")
                features_df = None
            if features_df is not None and "label" in features_df.columns:
                fmap = features_df.set_index("label")
                # Overwrite with freshly-computed features: the pipeline merges
                # them only *after* this hook, so any feature columns already on
                # `tracks` are the previous frame's (NaN for new cells).
                for col in fmap.columns:
                    if col == "label":
                        continue
                    current[col] = current["label"].map(fmap[col])
                    feature_cols.append(col)

        keep = [c for c in self._IDENTITY_COLS if c in current.columns]
        keep += [c for c in feature_cols if c not in keep]
        return current[keep]

    @staticmethod
    def _label_exposures(cells: pd.DataFrame, exposures: dict[int, float]) -> dict[int, float]:
        """Map server per-``particle`` exposures back onto per-``label`` exposures."""
        label_exposures: dict[int, float] = {}
        for _, row in cells.iterrows():
            particle = row.get("particle")
            label = row.get("label")
            if pd.isna(particle) or pd.isna(label):
                continue
            exposure = exposures.get(int(particle))
            if exposure is None:
                continue
            label_exposures[int(label)] = float(exposure)
        return label_exposures
