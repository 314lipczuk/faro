"""Replay a recorded experiment through the per-cell inference-dosing feature.

`ControllerSimulated` feeds previously-acquired frames from disk through the
*real* pipeline (segment -> track -> features), with `InferenceServerStim` as the
stimulator, so per-cell stim doses are computed on recorded data — no camera,
no hardware, no Micro-Manager install. This is the scaffold for testing the
feature (and, with --server, the real inference server) against real frames.

Run from the faro repo root, with faro's environment:

    python experiments/24_inference_server_stim/replay_inference.py
    python experiments/24_inference_server_stim/replay_inference.py \
        --data path/to/your_experiment --server http://localhost:8080 --cellpose

Data layout expected at --data: an `acquisition.ome.zarr/` (or `raw/` TIFFs)
plus the channels used below. Defaults target experiments/99_demo_data.

Observability: the inference client is wrapped to record every per-cell dose it
returns. On real hardware those become DMD staircase sub-frames; here we inspect
the decisions directly (the SLM display path needs real/`MMDemo` hardware).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make the repo (and its test helpers) importable when run as a script.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from faro.core.controller import ControllerSimulated
from faro.core.data_structures import Channel, RTMSequence, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.feature_extraction.erk_ktr import FE_ErkKtr
from faro.tracking.trackpy import TrackerTrackpy
from faro.inference.client import FakeInferenceClient, HttpInferenceClient
from faro.stimulation.inference_server import InferenceServerStim

from tests.fake_microscope import FakeMicroscope

DEFAULT_DATA = os.path.join(REPO, "experiments", "99_demo_data", "full_fov_stim")


class RecordingClient:
    """Wrap any InferenceClient and log the per-cell doses it returns."""

    def __init__(self, inner, eps_ms):
        self.inner = inner
        self.eps_ms = eps_ms
        self.log: list[dict] = []

    def predict(self, cells, meta):
        out = self.inner.predict(cells, meta)
        exp = np.array(list(out.values()), dtype=float)
        pos = exp[exp > 0]
        self.log.append({
            "fov": meta.get("fov"),
            "timestep": meta.get("timestep"),
            "n_cells": len(out),
            "min_ms": round(float(pos.min()), 1) if len(pos) else 0.0,
            "max_ms": round(float(pos.max()), 1) if len(pos) else 0.0,
            "mean_ms": round(float(pos.mean()), 1) if len(pos) else 0.0,
            # DMD staircase steps == distinct dose levels on the eps_ms grid
            "n_subframes": int(len(np.unique(np.round(pos / self.eps_ms)))),
        })
        return out


class ReplayScene:
    """Headless mic: camera frames ignored (disk frames used); SimDMD only."""
    image_height = image_width = 1024
    channels = ("miRFP", "mScarlet3", "stim-405")
    slm_name = "SLM"
    slm_shape = (1024, 1024)

    def render(self, event):
        return np.zeros((self.image_height, self.image_width), np.uint16)


def build_events(loops, positions, stim_frames):
    imaging = (
        Channel(config="miRFP", exposure=100, group="Channel"),      # ch0 (segment)
        Channel(config="mScarlet3", exposure=100, group="Channel"),  # ch1 (biosensor)
    )
    stim_ch = (Channel(config="stim-405", exposure=100, group="Channel"),)
    seq = RTMSequence(
        time_plan={"interval": 5.0, "loops": loops},
        stage_positions=[{"x": float(100 * p), "y": 0.0, "z": 0.0} for p in range(positions)],
        channels=imaging,
        stim_channels=stim_ch,
        stim_frames=list(stim_frames),
        rtm_metadata={"phase_name": "replay", "phase_id": 1, "treatment_name": "replay"},
    )
    return list(seq)


def build_segmentator(use_cellpose):
    if use_cellpose:
        from faro.segmentation.cellpose_v4 import CellposeV4
        return CellposeV4()
    from faro.segmentation.base import OtsuSegmentator
    return OtsuSegmentator()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA, help="recorded experiment dir")
    ap.add_argument("--server", default=None, help="inference server URL (else a fake rule)")
    ap.add_argument("--cellpose", action="store_true", help="use CellposeV4 (else Otsu)")
    ap.add_argument("--loops", type=int, default=3, help="timepoints to replay")
    ap.add_argument("--positions", type=int, default=2, help="FOVs to replay")
    ap.add_argument("--stim-frames", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--eps-ms", type=float, default=25.0)
    args = ap.parse_args()

    out = os.path.join(tempfile.gettempdir(), "faro-replay-inference")

    inner = (
        HttpInferenceClient(args.server) if args.server
        else FakeInferenceClient(rule=lambda c: float(np.clip(c.get("cnr", 0.0) * 800.0, 0, 3000)))
    )
    client = RecordingClient(inner, args.eps_ms)
    fe = FE_ErkKtr("labels")
    pipeline = ImageProcessingPipeline(
        storage_path=out,
        segmentators=[SegmentationMethod("labels", build_segmentator(args.cellpose), 0, True)],
        tracker=TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=fe,
        stimulator=InferenceServerStim(client, feature_extractor=fe, eps_ms=args.eps_ms),
    )
    ctrl = ControllerSimulated(
        FakeMicroscope(ReplayScene()), pipeline,
        old_data_project_path=args.data, writer=OmeZarrWriter(storage_path=out),
    )
    ctrl.run_experiment(
        build_events(args.loops, args.positions, args.stim_frames),
        stim_mode="current", validate=False,
    ).wait(300)
    ctrl.finish_experiment()

    print("\n=== background errors ===", ctrl.background_errors)
    if not client.log:
        print("No stim frames produced doses — check --stim-frames / segmentation.")
        return
    df = pd.DataFrame(client.log).sort_values(["fov", "timestep"])
    print("=== per-cell doses decided per stim frame ===")
    print(df.to_string(index=False))
    print(f"\nsource: {args.data}")
    print("n_subframes = distinct DMD staircase steps; DMD wall-clock ~ max_ms (not sum).")


if __name__ == "__main__":
    main()
