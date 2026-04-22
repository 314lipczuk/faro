"""Tests for ``faro.core.writers``.

Exercises the writer API directly (no Controller, no microscope), with a
focus on schema invariants that downstream analysis scripts depend on:

* TiffWriter: filename convention (``<folder>/<fname>.tiff``), lazy
  folder creation, round-trip of written bytes.
* OmeZarrWriter single-position: ome-writers stream path, label-group
  creation, stim-channel routing.
* OmeZarrWriter multi-position: direct store path, ``(t, p, c, y, x)``
  axis layout, multiscales metadata.
* OmeZarrWriterPlate: plate layout with rows/columns/wells.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from faro.core.data_structures import Channel, RTMEvent
from faro.core.writers import (
    OmeZarrWriter,
    OmeZarrWriterPlate,
    TiffWriter,
)


IMG_H, IMG_W = 64, 64
N_T = 4
IMG_CHANNELS = ["phase-contrast", "fitc"]
STIM_CHANNELS = 1


def _raw(t: int, p: int, *, n_channels: int = len(IMG_CHANNELS)) -> np.ndarray:
    """Multi-channel raw frame tagged with ``t, p`` for identifiability."""
    arr = np.zeros((n_channels, IMG_H, IMG_W), dtype=np.uint16)
    arr[..., t % IMG_H, p % IMG_W] = 50_000 + t * 100 + p
    return arr


def _mask(t: int, p: int) -> np.ndarray:
    """Simple 2D label image."""
    arr = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    arr[10:20, 10:20] = 1
    arr[30:40, 30:40] = 2
    return arr


def _meta(t: int, p: int, *, stim: bool = False) -> dict:
    return {
        "timestep": t,
        "fov": p,
        "stim": stim,
        "fname": f"{p:03d}_{t:05d}",
    }


# ===========================================================================
# TiffWriter
# ===========================================================================


class TestTiffWriter:
    """TIFF writer: one file per frame, lazy folder creation."""

    def test_write_creates_file_at_expected_path(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        img = _mask(0, 0)
        writer.write(img, _meta(0, 0), "labels")
        assert os.path.exists(os.path.join(tmp_dir, "labels", "000_00000.tiff"))

    def test_write_is_roundtrippable(self, tmp_dir):
        """Read-back must equal the original array."""
        writer = TiffWriter(tmp_dir)
        img = _mask(2, 1)
        writer.write(img, _meta(2, 1), "labels")
        roundtripped = tifffile.imread(
            os.path.join(tmp_dir, "labels", "001_00002.tiff")
        )
        np.testing.assert_array_equal(roundtripped, img)

    def test_auto_creates_folders(self, tmp_dir):
        """Folders are created lazily per-write with ``exist_ok``."""
        writer = TiffWriter(tmp_dir)
        for folder in ("raw", "labels", "stim_mask", "particles"):
            writer.write(_mask(0, 0), _meta(0, 0), folder)
            assert os.path.isdir(os.path.join(tmp_dir, folder))

    def test_save_events_produces_events_json(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        events = [
            RTMEvent(
                index={"t": t, "p": 0},
                channels=(Channel("phase-contrast", 50),),
            )
            for t in range(3)
        ]
        writer.save_events(events)
        assert os.path.exists(os.path.join(tmp_dir, "events.json"))

    def test_close_is_noop(self, tmp_dir):
        """TiffWriter has no buffers; close must not fail."""
        TiffWriter(tmp_dir).close()


# ===========================================================================
# OmeZarrWriter: single-position
# ===========================================================================


ZARR_DIRNAME = "acquisition.ome.zarr"


def _write_full_run(writer, *, n_pos: int, n_t: int = N_T) -> None:
    """Write a full t×p×{raw,stim,labels} run through ``writer``."""
    for t in range(n_t):
        for p in range(n_pos):
            writer.write(_raw(t, p), _meta(t, p), "raw")
            if t % 2 == 0:
                writer.write(
                    np.zeros((IMG_H, IMG_W), dtype=np.uint16),
                    _meta(t, p, stim=True),
                    "stim",
                )
            writer.write(_mask(t, p), _meta(t, p), "labels")


class TestOmeZarrWriterSinglePosition:
    """Single-FOV uses the ome-writers stream path."""

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=["Pos0"],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        _write_full_run(writer, n_pos=1)
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_store_exists_and_reopens(self, zarr_path):
        assert zarr_path.is_dir()
        zarr.open_group(str(zarr_path), mode="r")

    def test_raw_array_shape_and_dtype(self, zarr_path):
        """Single-pos layout exposes raw as ``0`` with (t, c, y, x)."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = root["0"]
        # imaging channels + stim channels
        assert raw.shape == (N_T, len(IMG_CHANNELS) + STIM_CHANNELS, IMG_H, IMG_W)
        assert str(raw.dtype) == "uint16"

    def test_labels_group_populated(self, zarr_path):
        """Labels are written as an NGFF label group at the root."""
        root = zarr.open_group(str(zarr_path), mode="r")
        labels_arr = root["labels/labels/0"]
        assert labels_arr.shape[-2:] == (IMG_H, IMG_W)


# ===========================================================================
# OmeZarrWriter: multi-position (direct path)
# ===========================================================================


class TestOmeZarrWriterMultiPosition:
    """Multi-FOV uses the direct-zarr path: single 5D array with a p axis."""

    N_POS = 3

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        _write_full_run(writer, n_pos=self.N_POS)
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_raw_has_position_axis(self, zarr_path):
        """Shape is ``(t, p, c, y, x)`` in direct mode."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = root["0"]
        assert raw.shape == (
            N_T,
            self.N_POS,
            len(IMG_CHANNELS) + STIM_CHANNELS,
            IMG_H,
            IMG_W,
        )

    def test_multiscales_axis_names(self, zarr_path):
        """NGFF metadata advertises the correct axis order."""
        root = zarr.open_group(str(zarr_path), mode="r")
        ome = root.attrs["ome"]
        axes = [a["name"] for a in ome["multiscales"][0]["axes"]]
        assert axes == ["t", "p", "c", "y", "x"]

    def test_raw_content_identifiable(self, zarr_path):
        """Raw array preserves the per-(t, p) tag we wrote."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = np.asarray(root["0"])
        for t in range(N_T):
            for p in range(self.N_POS):
                expected = 50_000 + t * 100 + p
                assert raw[t, p, 0, t % IMG_H, p % IMG_W] == expected


# ===========================================================================
# OmeZarrWriter: stim routing
# ===========================================================================


class TestOmeZarrStimRouting:
    """``store_stim_images=False`` keeps stim readouts out of the raw array."""

    def test_stim_disabled_excludes_stim_channels(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=["Pos0"],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        for t in range(N_T):
            writer.write(_raw(t, 0), _meta(t, 0), "raw")
        writer.close()

        root = zarr.open_group(
            str(Path(tmp_dir) / ZARR_DIRNAME), mode="r"
        )
        raw = root["0"]
        # Only the imaging channels — stim channels excluded
        assert raw.shape[-3] == len(IMG_CHANNELS)


# ===========================================================================
# OmeZarrWriterPlate
# ===========================================================================


class TestOmeZarrWriterPlate:
    """Plate writer lays positions out as a single-row HCS plate."""

    N_POS = 3

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriterPlate(
            tmp_dir, store_stim_images=False, n_timepoints=N_T
        )
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        for t in range(N_T):
            for p in range(self.N_POS):
                writer.write(_raw(t, p), _meta(t, p), "raw")
                writer.write(_mask(t, p), _meta(t, p), "labels")
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_plate_metadata_defines_one_well_per_position(self, zarr_path):
        """Each position becomes a well in row ``A``."""
        root = zarr.open_group(str(zarr_path), mode="r")
        plate = root.attrs["ome"]["plate"]
        assert [r["name"] for r in plate["rows"]] == ["A"]
        cols = [c["name"] for c in plate["columns"]]
        assert cols == [str(i + 1) for i in range(self.N_POS)]
        well_paths = sorted(w["path"] for w in plate["wells"])
        assert well_paths == [f"A/{i + 1}" for i in range(self.N_POS)]

    def test_well_image_array_shape(self, zarr_path):
        """Each well hosts a single image group ``<row>/<col>/0`` whose ``0``
        array carries the (t, c, y, x) raw stack."""
        for i in range(self.N_POS):
            arr = zarr.open_array(str(zarr_path / "A" / str(i + 1) / "0" / "0"), mode="r")
            assert arr.shape == (N_T, len(IMG_CHANNELS), IMG_H, IMG_W)
