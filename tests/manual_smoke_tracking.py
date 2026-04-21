"""Run the synthetic-cell pipeline and leave the outputs on disk for napari.

Reuses :class:`SyntheticCellScene`, :class:`StimPerCellCenter`, and the
helpers from :mod:`tests.test_tracking_accuracy`: same pipeline as
``test_tracker_recovers_gt_trajectories`` and
``test_stim_masks_centered_on_cells``, but written to a persistent temp
dir so you can load the raw frames, labels, stim masks, and tracks
parquet in napari.

Not a pytest file; run directly::

    python -m tests.manual_smoke_tracking
    python -m tests.manual_smoke_tracking --napari
    python -m tests.manual_smoke_tracking --writer ome-zarr --napari
    python -m tests.manual_smoke_tracking --output /path/to/dir

TiffWriter layout::

    <output>/raw/           raw synthetic frames
    <output>/labels/        segmentation masks
    <output>/stim_mask/     stim masks (one per stim frame)
    <output>/tracks/        parquet with particle/timestep/x/y columns

OmeZarrWriter layout (single store with multiscale + labels groups)::

    <output>/acquisition.ome.zarr/
    <output>/tracks/0_latest.parquet
    <output>/gt_positions.parquet
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import tifffile

from faro.core.controller import Controller
from faro.core.data_structures import SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter, TiffWriter
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import SegmentatorBinary
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.test_tracking_accuracy import (
    MEMORY,
    N_CELLS,
    N_FRAMES,
    SEARCH_RANGE,
    StimPerCellCenter,
    SyntheticCellScene,
    _make_events,
)

_ZARR_DIRNAME = "acquisition.ome.zarr"


def _make_writer(kind: str, output: str, *, n_frames: int, stim: bool):
    """Build the requested writer. FakeMicroscope speaks the full
    pymmcore-plus API (via FakeCMMCorePlus), so OmeZarrWriter's auto-init
    path through the Controller works without any special-casing.
    """
    if kind == "tiff":
        return TiffWriter(storage_path=output)
    if kind == "ome-zarr":
        return OmeZarrWriter(
            storage_path=output,
            store_stim_images=stim,
            n_timepoints=n_frames,
        )
    raise ValueError(f"Unknown writer kind: {kind!r}")


def run(
    output: str,
    *,
    n_frames: int = N_FRAMES,
    stim: bool = True,
    writer_kind: str = "tiff",
) -> str:
    os.makedirs(output, exist_ok=True)

    stimulator = StimPerCellCenter(radius=2) if stim else None
    stim_frames = tuple(range(1, n_frames)) if stim else ()

    pipeline = ImageProcessingPipeline(
        storage_path=output,
        segmentators=[SegmentationMethod("labels", SegmentatorBinary(), 0, True)],
        tracker=TrackerTrackpy(search_range=SEARCH_RANGE, memory=MEMORY),
        feature_extractor=SimpleFE("labels"),
        stimulator=stimulator,
    )
    scene = SyntheticCellScene(n_cells=N_CELLS, n_frames=n_frames, with_slm=stim)
    mic = FakeMicroscope(scene)
    writer = _make_writer(writer_kind, output, n_frames=n_frames, stim=stim)
    ctrl = Controller(mic, pipeline, writer=writer)
    ctrl.run_experiment(
        _make_events(n_frames, stim_frames=stim_frames),
        stim_mode="current",
        validate=False,
    )
    ctrl._analyzer.wait_idle()
    ctrl.finish_experiment()

    # Dump GT positions next to the other outputs so they can be
    # cross-referenced without re-running the generator.
    gt_rows = [
        {"timestep": t, "gt_particle": i, "x": r, "y": c}
        for t, frame in enumerate(scene.gt)
        for i, (r, c) in enumerate(frame)
    ]
    pd.DataFrame(gt_rows).to_parquet(os.path.join(output, "gt_positions.parquet"))

    return output


def _load_stack(folder: str) -> np.ndarray | None:
    if not os.path.isdir(folder):
        return None
    files = sorted(f for f in os.listdir(folder) if f.endswith(".tiff"))
    if not files:
        return None
    return np.stack([tifffile.imread(os.path.join(folder, f)) for f in files])


def _open_in_napari(output: str) -> None:
    import napari

    viewer = napari.Viewer()

    zarr_path = os.path.join(output, _ZARR_DIRNAME)
    if os.path.isdir(zarr_path):
        # OmeZarrWriter: let napari-ome-zarr surface images + labels.
        viewer.open(zarr_path, plugin="napari-ome-zarr")
    else:
        raw = _load_stack(os.path.join(output, "raw"))
        labels = _load_stack(os.path.join(output, "labels"))
        particles = _load_stack(os.path.join(output, "particles"))
        stim = _load_stack(os.path.join(output, "stim_mask"))
        if raw is not None:
            # Per-frame raw TIFFs are (C, Y, X) so the stack is (T, C, Y, X).
            # Tell napari the channel axis so it doesn't add a second slider.
            viewer.add_image(raw, name="raw", channel_axis=1)
        if labels is not None:
            viewer.add_labels(labels.astype(np.uint32), name="labels", visible=False)
        if particles is not None:
            # save_tracked=True writes particle IDs into pixel values, so
            # each cell keeps its colour across time.
            viewer.add_labels(particles.astype(np.uint32), name="particles")
        if stim is not None:
            viewer.add_image(
                stim, name="stim_mask", colormap="red", blending="additive"
            )

    tracks_parquet = os.path.join(output, "tracks", "0_latest.parquet")
    gt_parquet = os.path.join(output, "gt_positions.parquet")
    if os.path.exists(tracks_parquet):
        df = pd.read_parquet(tracks_parquet).sort_values(["particle", "timestep"])
        # napari tracks layer wants (track_id, t, y, x); the feature
        # extractor stores centroid-0 as "x" (row) and centroid-1 as
        # "y" (col) — napari's y/x match that (y=row, x=col).
        arr = df[["particle", "timestep", "x", "y"]].to_numpy()
        viewer.add_tracks(arr, name="tracks", tail_length=N_FRAMES)

    if os.path.exists(gt_parquet):
        gt = pd.read_parquet(gt_parquet).sort_values(["gt_particle", "timestep"])
        arr = gt[["gt_particle", "timestep", "x", "y"]].to_numpy()
        viewer.add_tracks(arr, name="gt", tail_length=N_FRAMES)

    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to write outputs to (default: a persistent tempdir)",
    )
    parser.add_argument("--napari", action="store_true", help="Open outputs in napari")
    parser.add_argument("--no-stim", action="store_true", help="Skip stim pass")
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument(
        "--writer",
        choices=("tiff", "ome-zarr"),
        default="tiff",
        help="Output format (default: tiff)",
    )
    args = parser.parse_args()

    output = args.output or tempfile.mkdtemp(prefix="faro_tracking_demo_")
    path = run(
        output,
        n_frames=args.frames,
        stim=not args.no_stim,
        writer_kind=args.writer,
    )
    print(f"Wrote outputs to: {path}")
    if args.writer == "ome-zarr":
        print(f"  ome-zarr:     {path}/{_ZARR_DIRNAME}")
    else:
        print(f"  raw frames:   {path}/raw/")
        print(f"  labels:       {path}/labels/")
        print(f"  particles:    {path}/particles/")
        print(f"  stim masks:   {path}/stim_mask/")
    print(f"  tracks:       {path}/tracks/0_latest.parquet")
    print(f"  GT positions: {path}/gt_positions.parquet")

    if args.napari:
        _open_in_napari(path)


if __name__ == "__main__":
    main()
