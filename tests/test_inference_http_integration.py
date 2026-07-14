"""Real-HTTP integration for the inference-server dosing path.

Unlike ``test_pipeline_stim_inference`` (which uses the in-process fake), this
spins up an actual HTTP server implementing the ``/predict`` contract and drives
the full closed loop through :class:`HttpInferenceClient` — exercising JSON
serialization, the socket round-trip, and response parsing end-to-end. The stub
server is launched by the test itself, so it always runs in CI with no external
dependency.

To smoke-test against a *real* server (e.g. the cluster model over an SSH
tunnel) set ``FARO_INFERENCE_SERVER_URL``; that test is skipped otherwise.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from faro.core.controller import Controller
from faro.feature_extraction.simple import SimpleFE
from faro.inference.client import HttpInferenceClient
from faro.stimulation.inference_server import InferenceServerStim

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    make_events,
    make_pipeline as _make_pipeline,
    run_and_wait,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)


# ===================================================================
# A minimal stdlib /predict server implementing the faro contract
# ===================================================================


def _make_handler(dose_fn):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            exposures = {}
            for cell in payload.get("cells", []):
                particle = cell.get("particle")
                if particle is None:
                    continue
                exposures[str(int(particle))] = float(dose_fn(cell))
            body = json.dumps(
                {
                    "fov": payload.get("fov"),
                    "timestep": payload.get("timestep"),
                    "exposures": exposures,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler


class _StubServer:
    """Context-managed background HTTP server on an ephemeral port."""

    def __init__(self, dose_fn):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(dose_fn))
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _area_dose(cell):
    # bigger cell -> 100 ms, smaller -> 250 ms (feature computed by the stimulator)
    return 100.0 if cell.get("area", 0) > 900 else 250.0


# ===================================================================
# HttpInferenceClient against the real stub
# ===================================================================


def test_http_client_roundtrip():
    with _StubServer(_area_dose) as srv:
        client = HttpInferenceClient(srv.url)
        cells = pd.DataFrame({"particle": [1, 2], "area": [1200.0, 700.0]})
        out = client.predict(cells, {"fov": 0, "timestep": 0, "time": 0.0})
    assert out == {1: 100.0, 2: 250.0}


def test_http_client_retries_then_fails_fast():
    # No server listening on this port -> retries exhausted -> RuntimeError.
    client = HttpInferenceClient(
        "http://127.0.0.1:9", max_attempts=2, retry_backoff=0.01, timeout=0.2
    )
    cells = pd.DataFrame({"particle": [1], "area": [1000.0]})
    with pytest.raises(RuntimeError):
        client.predict(cells, {"fov": 0, "timestep": 0})


# ===================================================================
# Full closed loop over HTTP
# ===================================================================


class TestClosedLoopOverHttp:
    STIM_FRAMES = (2, 3)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self._srv = _StubServer(_area_dose)
        self._srv.__enter__()
        client = HttpInferenceClient(self._srv.url)
        stim = InferenceServerStim(
            client, feature_extractor=SimpleFE("labels"), eps_ms=25.0
        )
        pipeline = _make_pipeline(self.path, tracker=tracker, stimulator=stim)
        self.scene = _RecScene()
        self.ctrl = Controller(FakeMicroscope(self.scene), pipeline)
        run_and_wait(
            self.ctrl, make_events(5, stim_frames=self.STIM_FRAMES), stim_mode="current"
        )
        yield
        self._srv.__exit__(None, None, None)

    def test_staircase_fired_over_http(self):
        by_t: dict[int, list[float]] = {}
        for t, _px, exposure in self.scene.records:
            by_t.setdefault(t, []).append(exposure)
        assert set(by_t) == set(self.STIM_FRAMES)
        for t in self.STIM_FRAMES:
            assert by_t[t] == [100.0, 150.0], f"frame {t}: {by_t[t]}"
        assert self.ctrl.background_errors == []


class _RecScene(CircleScene):
    def __init__(self):
        super().__init__(with_slm=True)
        self.records: list[tuple[int, int, float]] = []

    def on_slm_displayed(self, image, event):
        super().on_slm_displayed(image, event)
        self.records.append(
            (event.index.get("t", 0), int((np.asarray(image) > 0).sum()), event.exposure)
        )


# ===================================================================
# Optional smoke test against a real server (skipped unless configured)
# ===================================================================


@pytest.mark.skipif(
    not os.environ.get("FARO_INFERENCE_SERVER_URL"),
    reason="set FARO_INFERENCE_SERVER_URL to smoke-test a real inference server",
)
def test_real_server_smoke():
    client = HttpInferenceClient(os.environ["FARO_INFERENCE_SERVER_URL"])
    cells = pd.DataFrame(
        {"particle": [1, 2], "x": [10.0, 20.0], "y": [10.0, 20.0], "cnr": [0.8, 1.6]}
    )
    out = client.predict(cells, {"fov": 0, "timestep": 0, "time": 0.0})
    assert isinstance(out, dict)
    for particle, exposure in out.items():
        assert isinstance(particle, int)
        assert 0.0 <= exposure <= 3000.0
