"""Status reporting + cancellation handle for asynchronous experiment runs.

``Controller.run_experiment`` returns a ``RunHandle``. The handle owns:

* the worker thread driving the MDA feed loop,
* a cooperative cancellation event,
* an immutable ``RunStatus`` snapshot that the controller's threads update
  via :meth:`RunHandle.update`,
* a ``statusChanged`` :mod:`psygnal` signal that UI widgets (and any other
  observer) can subscribe to for live updates.

The handle is thread-safe: callers and the worker can both read ``status()``
and the worker can call ``update(...)`` without coordination from the
caller. Status updates emit ``statusChanged`` synchronously; with ``qtpy``
loaded :mod:`psygnal` routes the emission to listeners' threads through
Qt's queued connections, so a napari widget connected on the main thread
sees a queued slot call from the worker without extra plumbing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from psygnal import Signal

if TYPE_CHECKING:
    pass


RunState = Literal["pending", "running", "cancelling", "done", "error"]


@dataclass(frozen=True)
class RunStatus:
    """Immutable snapshot of an experiment run."""

    state: RunState = "pending"
    # Latest RTMEvent index the feed loop committed to the MDA queue.
    current_event_index: dict[str, int] | None = None
    current_fov: int | None = None
    # Counts.
    n_events_total: int = 0          # how many RTMEvents the run was started with
    n_events_consumed: int = 0       # RTMEvents pulled by the feed loop so far
    n_frames_received: int = 0       # frames acknowledged via frameReady
    # Timing.
    started_at: float | None = None       # time.monotonic() when the worker began
    finished_at: float | None = None      # time.monotonic() when the worker exited
    last_frame_wallclock: float | None = None
    # Lag: (time.monotonic() - started_at) - event.min_start_time, in ms,
    # at the most recent frame_ready. Positive == we're behind schedule.
    lag_ms: float | None = None
    # Pipeline / storage backpressure visibility (best-effort).
    pipeline_inflight: int = 0
    storage_queue_depth: int = 0
    # Errors. ``background_errors`` accumulates analyzer-side issues; ``fatal_error``
    # is set when the worker itself raises.
    background_errors: tuple[Any, ...] = field(default_factory=tuple)
    fatal_error: BaseException | None = None


class RunHandle:
    """Handle returned by ``Controller.run_experiment`` / ``continue_experiment``.

    Use ``wait()`` to block until the run finishes, ``cancel()`` to request a
    graceful stop, ``status()`` for a snapshot, or subscribe to
    ``statusChanged`` for live updates. ``statusChanged`` is a per-instance
    :mod:`psygnal` signal that emits the latest ``RunStatus`` after every
    update.
    """

    # psygnal class-level Signal is a descriptor; access via ``handle.statusChanged``
    # gives an instance-bound signal. Different handles have independent signals.
    statusChanged = Signal(RunStatus)

    def __init__(self, n_events_total: int = 0) -> None:
        self._lock = threading.RLock()
        self._status = RunStatus(state="pending", n_events_total=n_events_total)
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public API ---------------------------------------------------------

    def status(self) -> RunStatus:
        """Return the current immutable status snapshot."""
        with self._lock:
            return self._status

    def wait(self, timeout: float | None = None) -> RunStatus:
        """Block until the worker thread finishes (or ``timeout`` elapses).

        Returns the final ``RunStatus``. Re-raises ``fatal_error`` if the
        worker crashed -- mirroring the previous synchronous-run behaviour.
        """
        if self._thread is not None:
            self._thread.join(timeout)
        status = self.status()
        if status.fatal_error is not None:
            raise status.fatal_error
        return status

    def cancel(self) -> None:
        """Request graceful cancellation. Idempotent. Does not block.

        Sets the cancel event the feed loop polls on each iteration; on the
        next poll the loop stops feeding new events, asks the MDA engine to
        abort the in-flight event, and exits. Use ``wait()`` afterwards to
        block until the worker actually stops.
        """
        self._cancel_event.set()
        with self._lock:
            if self._status.state == "running":
                self._status = replace(self._status, state="cancelling")
                new_status = self._status
            else:
                return
        self.statusChanged.emit(new_status)

    def is_running(self) -> bool:
        """True if the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # -- worker-side helpers ------------------------------------------------

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative-cancel event the feed loop polls. Worker-side."""
        return self._cancel_event

    def update(self, **updates: Any) -> RunStatus:
        """Atomically apply ``updates`` to the status snapshot and emit.

        Called from worker / pipeline / storage threads. ``statusChanged``
        listeners on the main thread see queued-connection delivery via
        psygnal's Qt integration.
        """
        with self._lock:
            self._status = replace(self._status, **updates)
            new_status = self._status
        self.statusChanged.emit(new_status)
        return new_status
