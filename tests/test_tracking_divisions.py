"""Tracking behavior around cell divisions.

One cell at t=0-2 splits into two daughters at t=3 and both daughters
continue through t=5. Runs the full pipeline for both Trackpy and
Motile, and verifies tracker-agnostic invariants:

* Both daughters are tracked in every post-division frame.
* Each daughter has a single particle ID across post-division frames.
* Pre-division frames see exactly one particle; post-division frames
  see exactly two.

Motile models divisions natively (both daughters get freshly-allocated
IDs distinct from the parent); Trackpy's subnet linking assigns a new
ID to one daughter and keeps the parent's ID on the other. The test
accepts either behavior.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from useq import MDAEvent

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import SegmentatorBinary
from faro.tracking.motile_tracker import TrackerMotile
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope


IMG_SIZE = 128
CELL_RADIUS = 4
CELL_VALUE = 50_000
DIVISION_FRAME = 3
N_FRAMES = 6
DAUGHTER_SEPARATION = 12  # px between the two daughters after division


def _render_disks(positions: np.ndarray) -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    for r, c in positions:
        img[(yy - r) ** 2 + (xx - c) ** 2 <= CELL_RADIUS**2] = CELL_VALUE
    return img


class _DividingCellScene:
    """One cell for ``DIVISION_FRAME`` frames, then two daughters.

    Before division the parent sits at ``(cy, cx)``. From
    ``DIVISION_FRAME`` onwards the two daughters sit symmetrically
    ``DAUGHTER_SEPARATION / 2`` pixels either side of the parent along
    the column axis and drift outward by 1 px/frame.
    """

    image_height = IMG_SIZE
    image_width = IMG_SIZE
    channels = ("phase-contrast",)

    def __init__(self, cy: int = 64, cx: int = 64):
        self._cy = cy
        self._cx = cx

    def render(self, event: MDAEvent) -> np.ndarray:
        t = event.index.get("t", 0)
        if t < DIVISION_FRAME:
            return _render_disks(np.array([[self._cy, self._cx]]))
        drift = t - DIVISION_FRAME
        half = DAUGHTER_SEPARATION // 2
        return _render_disks(
            np.array(
                [
                    [self._cy, self._cx - half - drift],
                    [self._cy, self._cx + half + drift],
                ]
            )
        )


def _make_events(n_frames: int) -> list[RTMEvent]:
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=50),),
        )
        for t in range(n_frames)
    ]


@pytest.fixture(
    params=[
        pytest.param(TrackerTrackpy, id="Trackpy"),
        pytest.param(TrackerMotile, id="Motile"),
    ],
)
def tracker(request):
    # Generous search_range (> daughter drift per frame) and a memory
    # window so neither tracker hard-rejects the new detections.
    return request.param(search_range=30, memory=3)


def _run(tmp_dir: str, tracker) -> pd.DataFrame:
    pipeline = ImageProcessingPipeline(
        storage_path=tmp_dir,
        segmentators=[SegmentationMethod("labels", SegmentatorBinary(), 0, True)],
        tracker=tracker,
        feature_extractor=SimpleFE("labels"),
        stimulator=None,
    )
    scene = _DividingCellScene()
    mic = FakeMicroscope(scene)
    ctrl = Controller(mic, pipeline)
    ctrl.run_experiment(_make_events(N_FRAMES), validate=False)
    ctrl._analyzer.wait_idle()
    ctrl.finish_experiment()
    assert not ctrl.background_errors, ctrl.background_errors
    return pd.read_parquet(os.path.join(tmp_dir, "tracks", "0_latest.parquet"))


def test_division_produces_two_tracked_daughters(tmp_dir, tracker):
    df = _run(tmp_dir, tracker)

    # Pre-division: exactly one particle per frame.
    for t in range(DIVISION_FRAME):
        frame = df[df["timestep"] == t]
        assert len(frame) == 1, (
            f"pre-division frame t={t} has {len(frame)} detections"
        )

    # Post-division: exactly two detections per frame, each tagged with
    # a particle ID that persists for the rest of the run.
    daughter_ids_per_frame: list[set[int]] = []
    for t in range(DIVISION_FRAME, N_FRAMES):
        frame = df[df["timestep"] == t]
        assert len(frame) == 2, (
            f"post-division frame t={t} has {len(frame)} detections"
        )
        daughter_ids_per_frame.append(set(frame["particle"].tolist()))

    # Each daughter's ID holds across all post-division frames.
    persistent_ids = set.intersection(*daughter_ids_per_frame)
    assert len(persistent_ids) == 2, (
        f"Expected two particle IDs persisting through all post-division "
        f"frames; got {persistent_ids}. Per-frame: {daughter_ids_per_frame}"
    )
