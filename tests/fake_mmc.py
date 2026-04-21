"""Minimum viable pymmcore-plus ``CMMCorePlus`` fake for tests.

Implements only the slice of the CMMCorePlus surface that faro reaches
for during a test run (with ``validate=False``): ``run_mda``, the
``mda.events.frameReady`` signal, SLM dispatch, ``getImage{Height,Width}``,
and ``getChannelGroup``. Extend as tests need more.

Usage::

    class MyScene:
        image_height = 256
        image_width = 256
        slm_name = "FakeSLM"
        slm_shape = (256, 256)

        def render(self, event) -> np.ndarray: ...

    mmc = FakeCMMCorePlus(MyScene())
"""

from __future__ import annotations

import threading
from typing import Iterable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from psygnal import Signal
from useq import MDAEvent


@runtime_checkable
class Scene(Protocol):
    """Pluggable image source (and optional SLM recorder).

    Required: ``image_height`` / ``image_width`` plus ``render(event)``.
    Set ``slm_name`` and ``slm_shape`` to declare an SLM so
    ``FakeMicroscope`` attaches a DMD. Override ``channel_group`` to
    change the default name surfaced through ``getChannelGroup``.
    Implement ``on_slm_displayed(image, event)`` to record dispatched
    masks with event context.
    """

    image_height: int
    image_width: int
    channel_group: str
    slm_name: Optional[str]
    slm_shape: Optional[Tuple[int, int]]

    def render(self, event: MDAEvent) -> np.ndarray: ...


class _MDAEvents:
    frameReady = Signal(np.ndarray, MDAEvent)


class FakeMDA:
    """MDA loop that iterates events, dispatches SLM, emits ``frameReady``.

    Runs in a daemon thread. Stim events carrying ``slm_image`` are
    routed through ``setSLMImage``/``setSLMPixelsTo`` + ``displaySLMImage``
    so the mmc SLM path is exercised end-to-end; the scene's optional
    ``on_slm_displayed`` hook then fires with event context.
    """

    def __init__(self, mmc: "FakeCMMCorePlus") -> None:
        self._mmc = mmc
        self.events = _MDAEvents()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def cancel(self) -> None:
        self._cancel.set()

    def run(self, event_iter: Iterable[MDAEvent]) -> threading.Thread:
        self._cancel.clear()

        def _run() -> None:
            for event in event_iter:
                if self._cancel.is_set():
                    break
                self._dispatch_slm(event)
                img = self._mmc.scene.render(event)
                self.events.frameReady.emit(img, event)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self._thread

    def _dispatch_slm(self, event: MDAEvent) -> None:
        slm = getattr(event, "slm_image", None)
        if slm is None:
            return
        mmc = self._mmc
        if isinstance(slm.data, np.ndarray):
            mmc.setSLMImage(slm.device, slm.data)
        else:
            mmc.setSLMPixelsTo(slm.device, int(bool(slm.data)))
        mmc.displaySLMImage(slm.device)
        on_slm = getattr(mmc.scene, "on_slm_displayed", None)
        if on_slm is not None and mmc._slm_buf is not None:
            on_slm(mmc._slm_buf, event)


class FakeCMMCorePlus:
    """Minimum viable ``CMMCorePlus`` fake."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.mda = FakeMDA(self)
        self._slm_buf: Optional[np.ndarray] = None

    def run_mda(self, event_iter) -> threading.Thread:
        return self.mda.run(event_iter)

    def getImageHeight(self) -> int:
        return self.scene.image_height

    def getImageWidth(self) -> int:
        return self.scene.image_width

    def getChannelGroup(self) -> str:
        return getattr(self.scene, "channel_group", "Channel")

    def setSLMImage(self, name: str, image: np.ndarray) -> None:
        self._slm_buf = np.asarray(image).copy()

    def setSLMPixelsTo(self, name: str, value: int) -> None:
        h, w = self.scene.slm_shape
        self._slm_buf = np.full((h, w), int(value), dtype=np.uint8)

    def displaySLMImage(self, name: str) -> None:
        pass
