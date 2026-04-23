"""Failure-path and stress-test pipeline integration tests.

Covers:

* ``TestStressSlowSegmentation`` — 50-frame acquisition with slow
  segmentation exercises the deferred-queue backpressure path.
* ``TestCrashingSegmentator`` / ``Tracker`` / ``Stimulator`` /
  ``FeatureExtractor`` — each pipeline stage's fatal error path must
  still save raw images and leave downstream workers unblocked.
* ``TestBurstNoSignalLoss`` — 100-frame burst against a near-instant
  tiny pipeline; no frames should be dropped under back-pressure.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pytest
import tifffile

from faro.core.controller import Controller
from faro.core.data_structures import SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    CrashingStimulator,
    make_events,
    run_and_wait,
    run_and_wait_long,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)


class SlowSegmentator(OtsuSegmentator):
    """OtsuSegmentator that sleeps *delay* seconds per frame to simulate load."""

    def __init__(self, delay: float = 1.0):
        super().__init__()
        self._delay = delay

    def segment(self, image: np.ndarray) -> np.ndarray:
        time.sleep(self._delay)
        return super().segment(image)


# ===================================================================
# Test Class 9: Stress test — slow segmentation, rapid acquisition
# ===================================================================


N_STRESS_FRAMES = 50
STRESS_SEG_DELAY = 0.1  # seconds per segmentation


class TestStressSlowSegmentation:
    """Acquire 20 frames with no delay while segmentation takes 1s each.

    Verifies that the deferred-queue mechanism in Analyzer eventually
    processes every frame: all raw images stored, all segmentation masks
    produced, all stim masks produced, and tracking covers every frame.
    """

    STIM_FRAMES = tuple(range(5, 15))  # stim on frames 5-14

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = ImageProcessingPipeline(
            storage_path=self.path,
            segmentators=[
                SegmentationMethod(
                    "labels", SlowSegmentator(STRESS_SEG_DELAY), 0, False
                )
            ],
            tracker=tracker,
            feature_extractor=SimpleFE("labels"),
            stimulator=CenterCircle(),
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_STRESS_FRAMES, stim_frames=self.STIM_FRAMES)
        run_and_wait_long(
            self.ctrl,
            self.events,
            stim_mode="current",
            timeout=N_STRESS_FRAMES * STRESS_SEG_DELAY + 60,
        )

    def test_all_raw_images_stored(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} raw TIFFs, got {len(files)}"

    def test_all_segmentation_masks_produced(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} label masks, got {len(files)}"

    def test_all_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} stim masks, got {len(files)}"

    def test_stim_masks_nonzero_on_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert mask.max() > 0, f"Frame {i}: stim frame should have nonzero mask"

    def test_tracking_covers_all_frames(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_STRESS_FRAMES
            ), f"Particle {pid}: expected {N_STRESS_FRAMES} rows, got {len(rows)}"

    def test_segmentation_quality_maintained(self):
        """Even under load, every frame should segment into exactly 2 labels."""
        labels_dir = os.path.join(self.path, "labels")
        files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".tiff")])
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            unique = set(np.unique(labels)) - {0}
            assert len(unique) == 2, f"{f}: expected 2 labels, got {len(unique)}"


# ===================================================================
# Failing component helpers
# ===================================================================

class CrashingSegmentator(OtsuSegmentator):
    """Segmentator that raises on every call."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError("Segmentation crashed!")


class CrashingTracker(TrackerTrackpy):
    """Tracker that raises on every call."""

    def track_cells(self, df_old, df_new, fov_state):
        raise RuntimeError("Tracking crashed!")

class CrashingFE(SimpleFE):
    """Feature extractor that raises on every call."""

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        raise RuntimeError("Feature extraction crashed!")


# ===================================================================
# Crash-test helpers
# ===================================================================
N_CRASH_FRAMES = 3
CRASH_QUEUE_TIMEOUT = 1  # seconds (instead of default 20)


def _make_crashing_pipeline(
    path, *, segmentator=None, tracker=None, fe=None, stim=None
):
    """Build a pipeline with short queue timeout for crash tests."""
    pipeline = ImageProcessingPipeline(
        storage_path=path,
        segmentators=[
            SegmentationMethod("labels", segmentator or OtsuSegmentator(), 0, False)
        ],
        tracker=tracker or TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=fe or SimpleFE("labels"),
        stimulator=stim,
    )
    pipeline._queue_timeout = CRASH_QUEUE_TIMEOUT
    return pipeline


# ===================================================================
# Test Class 10: Failing segmentator — raw images must still be saved
# ===================================================================


class TestCrashingSegmentator:
    """Pipeline crashes during segmentation. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            segmentator=CrashingSegmentator(),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        """Raw images are saved BEFORE pipeline.run(), so they must all exist."""
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks(self):
        """Pipeline crashed during segmentation, so no label masks should exist."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Test Class 11: Failing tracker — raw images saved
# ===================================================================


class TestCrashingTracker:
    """Pipeline crashes during tracking. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            tracker=CrashingTracker(search_range=50, memory=3),
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks_saved(self):
        """Crash happens after segmentation but before TIFF save (which is after put()).
        Since tracking crashes before put(), the TIFF saves never run."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Test Class 12: Failing stimulator — raw images saved
# ===================================================================


class TestCrashingStimulator:
    """Pipeline crashes during stim mask generation. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        # Stim on last frame only so no subsequent frame waits on the broken put()
        self.n_frames = N_CRASH_FRAMES
        self.stim_frames = (self.n_frames - 1,)
        self.pipeline = _make_crashing_pipeline(
            self.path,
            stim=CrashingStimulator(),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(self.n_frames, stim_frames=self.stim_frames)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == self.n_frames


# ===================================================================
# Test Class 13: Failing feature extractor — raw images saved
# ===================================================================


class TestCrashingFeatureExtractor:
    """Pipeline crashes during feature extraction. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            fe=CrashingFE("labels"),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks_saved(self):
        """FE crash happens before put() and TIFF saves, so no masks are written."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Continuation / extension helpers
# ===================================================================




# ===================================================================
# Test Class 14: Continue experiment — sequential continuation
# ===================================================================

N_BURST_FRAMES = 100
FIRST_FRAME_DELAY = 1.0
_BURST_IMG_SIZE = 16


def _make_tiny_image() -> np.ndarray:
    """10x10 uint16 image with two bright pixels (labels 1 and 2)."""
    img = np.zeros((_BURST_IMG_SIZE, _BURST_IMG_SIZE), dtype=np.uint16)
    img[3, 3] = 50000
    img[12, 12] = 50000
    return img


class _TinyScene:
    """16×16 frames with two bright pixels, used by the burst stress test."""

    image_height = 16
    image_width = 16
    channels = ("phase-contrast",)

    def render(self, event) -> np.ndarray:
        return _make_tiny_image()


class _FastSegmentator(OtsuSegmentator):
    """Near-instant segmentator: threshold + label.

    First call is delayed to create pipeline back-pressure.
    """

    def __init__(self, first_delay: float = FIRST_FRAME_DELAY):
        super().__init__()
        self._first_delay = first_delay
        self._lock = threading.Lock()
        self._first_done = False

    def segment(self, image: np.ndarray) -> np.ndarray:
        from skimage.measure import label

        with self._lock:
            is_first = not self._first_done
            self._first_done = True
        if is_first:
            time.sleep(self._first_delay)
        return label(image > image.mean())


class _FastTracker:
    """Tracker that assigns particle IDs by label without trackpy linking."""

    required_metadata: set[str] = set()

    def track_cells(self, df_old, df_new, fov_state):
        if "particle" not in df_new.columns:
            df_new = df_new.copy()
            df_new["particle"] = df_new["label"]
        return pd.concat([df_old, df_new], ignore_index=True)


class _FastFE:
    """Feature extractor returning positions and area via regionprops."""

    def __init__(self, used_mask):
        self.used_mask = used_mask

    def extract_positions(self, labels):
        from skimage.measure import regionprops_table

        table = regionprops_table(
            labels[self.used_mask], properties=["label", "centroid"]
        )
        df = pd.DataFrame.from_dict(table)
        df = df.rename({"centroid-0": "x", "centroid-1": "y"}, axis="columns")
        return df

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        from skimage.measure import regionprops_table

        table = regionprops_table(labels[self.used_mask], properties=["label", "area"])
        return pd.DataFrame.from_dict(table), None


@pytest.fixture(scope="module")
def burst_result():
    """Run the burst experiment once and share results across all tests."""
    path = tempfile.mkdtemp()
    pipeline = ImageProcessingPipeline(
        storage_path=path,
        segmentators=[
            SegmentationMethod("labels", _FastSegmentator(), 0, False),
        ],
        tracker=_FastTracker(),
        feature_extractor=_FastFE("labels"),
        stimulator=CenterCircle(),
    )
    pipeline._queue_timeout = 120
    mic = FakeMicroscope(_TinyScene())
    ctrl = Controller(mic, pipeline)
    stim_frames = tuple(range(10, 20))
    events = make_events(N_BURST_FRAMES, stim_frames=stim_frames)
    run_and_wait_long(
        ctrl,
        events,
        stim_mode="current",
        timeout=FIRST_FRAME_DELAY + N_BURST_FRAMES * 0.1 + 60,
    )
    yield path
    shutil.rmtree(path)


class TestBurstNoSignalLoss:
    """Dispatch 100 frames rapidly while the first frame blocks the pipeline.

    Uses tiny images (16x16) and fast fakes for segmentation/tracking/FE so
    the test exercises the controller's dispatch plumbing (queues, deferred
    worker, storage) rather than real image-analysis performance.
    """

    def test_all_raw_images_stored(self, burst_result):
        """Storage path: no images silently dropped."""
        raw_dir = os.path.join(burst_result, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} raw TIFFs, got {len(files)}"

    def test_all_segmentation_masks_produced(self, burst_result):
        """Pipeline path: every frame segmented despite initial block."""
        labels_dir = os.path.join(burst_result, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} label masks, got {len(files)}"

    def test_all_stim_masks_produced(self, burst_result):
        """Stim path: every frame produced a stim mask."""
        stim_dir = os.path.join(burst_result, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} stim masks, got {len(files)}"

    def test_tracking_covers_all_frames(self, burst_result):
        """Tracking: both particles tracked across every frame."""
        tracks_dir = os.path.join(burst_result, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_BURST_FRAMES
            ), f"Particle {pid}: expected {N_BURST_FRAMES} rows, got {len(rows)}"


# ===================================================================
# Test Class: Analyzer.get_stim_mask timeout recording
# ===================================================================

