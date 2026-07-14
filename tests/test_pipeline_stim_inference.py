"""Tests for the inference-server per-cell dosing path.

Two layers, no network:

* unit tests for the pure pieces — :func:`build_staircase` and
  :class:`FakeInferenceClient`;
* an end-to-end pipeline test driving :class:`InferenceServerStim` +
  :class:`FakeInferenceClient` on a :class:`FakeMicroscope`, asserting the
  controller emits one shrinking DMD sub-frame per staircase step with the
  right per-sub-frame exposure.

The real HTTP transport is covered separately (gated integration test); the
fake client exercises the same ``InferenceClient`` interface so a green run
here transfers to the real server modulo the network/model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from faro.core.controller import Controller
from faro.feature_extraction.simple import SimpleFE
from faro.inference.client import FakeInferenceClient, HttpInferenceClient
from faro.stimulation.inference_server import InferenceServerStim
from faro.stimulation.staircase import build_staircase

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    make_events,
    make_pipeline as _make_pipeline,
    run_and_wait,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)


# ===================================================================
# Unit: build_staircase
# ===================================================================


def _two_cell_labels():
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2
    return labels


class TestBuildStaircase:
    def test_cumulative_masks_and_durations(self):
        labels = _two_cell_labels()
        sub = build_staircase(labels, {1: 100.0, 2: 250.0}, eps_ms=25.0)
        assert [round(d, 3) for _, d in sub] == [100.0, 150.0]
        # sub 0 lights both cells; sub 1 lights only the 250 ms cell.
        assert sub[0][0][2, 2] == 1 and sub[0][0][7, 7] == 1
        assert sub[1][0][2, 2] == 0 and sub[1][0][7, 7] == 1
        # masks are cumulative (non-increasing area).
        assert sub[0][0].sum() > sub[1][0].sum() > 0

    def test_equal_exposures_collapse_to_one_subframe(self):
        labels = _two_cell_labels()
        sub = build_staircase(labels, {1: 120.0, 2: 130.0}, eps_ms=25.0)
        # 120 and 130 snap to the same 125 ms grid point -> one sub-frame.
        assert len(sub) == 1
        assert sub[0][0][2, 2] == 1 and sub[0][0][7, 7] == 1

    def test_no_positive_exposures_returns_empty(self):
        labels = _two_cell_labels()
        assert build_staircase(labels, {}, eps_ms=25.0) == []
        assert build_staircase(labels, {1: 0.0, 2: -3.0}, eps_ms=25.0) == []

    def test_small_exposure_not_dropped_by_snapping(self):
        labels = _two_cell_labels()
        # 5 ms would round to 0 on a 25 ms grid; must be floored to one level.
        sub = build_staircase(labels, {1: 5.0}, eps_ms=25.0)
        assert len(sub) == 1 and sub[0][1] > 0

    def test_max_subframes_caps_level_count(self):
        labels = np.zeros((30, 30), dtype=np.uint32)
        exposures = {}
        for i in range(1, 8):
            labels[i, :] = i
            exposures[i] = float(i * 13)
        sub = build_staircase(labels, exposures, eps_ms=1.0, max_subframes=3)
        assert 0 < len(sub) <= 3


# ===================================================================
# Unit: FakeInferenceClient
# ===================================================================


class TestFakeInferenceClient:
    def test_rule_applied_and_clipped(self):
        cells = pd.DataFrame({"particle": [10, 11], "cnr": [0.5, 100.0]})
        client = FakeInferenceClient(rule=lambda c: 200.0 * c["cnr"])
        out = client.predict(cells, {"fov": 0, "timestep": 1})
        assert out[10] == 100.0
        assert out[11] == 3000.0  # clipped to the default upper bound

    def test_empty_or_missing_particle_returns_empty(self):
        client = FakeInferenceClient(rule=lambda c: 100.0)
        assert client.predict(pd.DataFrame(), {}) == {}
        assert client.predict(pd.DataFrame({"cnr": [1.0]}), {}) == {}


# ===================================================================
# End-to-end: InferenceServerStim through the Controller
# ===================================================================


class _StaircaseScene(CircleScene):
    """CircleScene that also records each stim sub-frame's exposure."""

    def __init__(self, **kw):
        super().__init__(with_slm=True, **kw)
        # (timepoint, on_pixel_count, exposure_ms) per SLM display.
        self.records: list[tuple[int, int, float]] = []

    def on_slm_displayed(self, image, event) -> None:
        super().on_slm_displayed(image, event)
        on_pixels = int((np.asarray(image) > 0).sum())
        self.records.append((event.index.get("t", 0), on_pixels, event.exposure))


# Two circles of different area -> two distinct doses -> two staircase steps.
# Big cell (area > 900) = 100 ms, small cell = 250 ms.
def _area_rule(cell):
    return 100.0 if cell["area"] > 900 else 250.0


class TestInferenceServerStimCurrent:
    STIM_FRAMES = (2, 3)
    N_FRAMES = 5

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        client = FakeInferenceClient(rule=_area_rule)
        stim = InferenceServerStim(
            client, feature_extractor=SimpleFE("labels"), eps_ms=25.0
        )
        self.pipeline = _make_pipeline(self.path, tracker=tracker, stimulator=stim)
        self.scene = _StaircaseScene()
        self.ctrl = Controller(FakeMicroscope(self.scene), self.pipeline)
        events = make_events(self.N_FRAMES, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, events, stim_mode="current")

    def _records_by_t(self):
        by_t: dict[int, list[tuple[int, float]]] = {}
        for t, pixels, exposure in self.scene.records:
            by_t.setdefault(t, []).append((pixels, exposure))
        return by_t

    def test_two_subframes_per_stim_frame(self):
        by_t = self._records_by_t()
        assert set(by_t) == set(self.STIM_FRAMES)
        for t in self.STIM_FRAMES:
            assert len(by_t[t]) == 2, f"frame {t}: expected 2 DMD sub-frames"

    def test_subframe_exposures_are_threshold_deltas(self):
        # doses 100 & 250 ms -> thresholds [100, 250] -> durations [100, 150].
        by_t = self._records_by_t()
        for t in self.STIM_FRAMES:
            exposures = [exp for _, exp in by_t[t]]
            assert exposures == [100.0, 150.0], f"frame {t}: {exposures}"

    def test_subframe_masks_are_cumulative(self):
        by_t = self._records_by_t()
        for t in self.STIM_FRAMES:
            areas = [px for px, _ in by_t[t]]
            # first step lights both cells, second only the higher-dose cell.
            assert areas[0] > areas[1] > 0, f"frame {t}: {areas}"

    def test_stim_snaps_stored_distinctly(self):
        # Each sub-frame is its own IMG_STIM snap; disambiguated fnames
        # (``..._s{i}``) mean the K snaps per timepoint don't overwrite.
        stim_dir = os.path.join(self.path, "stim")
        files = set(os.listdir(stim_dir))
        for t in self.STIM_FRAMES:
            subs = [f for f in files if f.startswith(f"000_{t:05d}_s")]
            assert len(subs) == 2, f"frame {t}: expected 2 sub-frame snaps, got {subs}"

    def test_segmentation_intact_and_no_errors(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == self.N_FRAMES
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            assert len(set(np.unique(labels)) - {0}) == 2
        assert self.ctrl.background_errors == []


class TestInferenceServerStimNoStimWhenZeroDose:
    """A client returning 0 ms for every cell => no DMD sub-frames fire."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        client = FakeInferenceClient(rule=lambda c: 0.0)
        stim = InferenceServerStim(
            client, feature_extractor=SimpleFE("labels"), eps_ms=25.0
        )
        pipeline = _make_pipeline(self.path, tracker=tracker, stimulator=stim)
        self.scene = _StaircaseScene()
        self.ctrl = Controller(FakeMicroscope(self.scene), pipeline)
        events = make_events(4, stim_frames=(2, 3))
        run_and_wait(self.ctrl, events, stim_mode="current")

    def test_no_slm_events(self):
        assert self.scene.records == []
        assert self.ctrl.background_errors == []


# ===================================================================
# HttpInferenceClient: request/response mapping (no server, monkeypatched)
# ===================================================================


class TestHttpInferenceClientMapping:
    def test_serialize_and_parse(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"exposures": {"10": 120.0, "11": 9000.0}}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)

        client = HttpInferenceClient("http://cluster:8080")
        cells = pd.DataFrame({"particle": [10, 11], "cnr": [0.5, np.nan]})
        out = client.predict(cells, {"fov": 3, "timestep": 7, "time": 12.0})

        assert captured["url"] == "http://cluster:8080/predict"
        assert captured["json"]["fov"] == 3 and captured["json"]["timestep"] == 7
        # NaN feature serialized as null (JSON-safe).
        assert captured["json"]["cells"][1]["cnr"] is None
        # response parsed, clipped to [0, 3000].
        assert out == {10: 120.0, 11: 3000.0}
