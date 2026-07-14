"""Clients for a deep-learning inference server that returns per-cell stim doses.

faro extracts per-cell features from each acquired frame and asks an inference
server for a per-cell stimulation exposure (milliseconds). The transport is
swappable behind :class:`InferenceClient`:

- :class:`FakeInferenceClient` — in-process, no network; exposures from a rule.
- :class:`HttpInferenceClient` — POSTs features to an HTTP server, parses doses.
"""

from .client import FakeInferenceClient, HttpInferenceClient, InferenceClient

__all__ = ["InferenceClient", "FakeInferenceClient", "HttpInferenceClient"]
