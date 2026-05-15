"""Minimal Qt widget that mirrors a Controller's current ``RunHandle``.

Construct with a :class:`faro.core.controller.Controller` instance:

    from faro.widgets import ExperimentStatusWidget
    widget = ExperimentStatusWidget(ctrl)
    viewer.window.add_dock_widget(widget, name="Experiment")

The widget subscribes to ``ctrl.runStarted``, so it automatically re-binds
to whichever run is current. Each ``RunHandle.statusChanged`` emission
updates the labels and progress bar; the Stop button calls
``handle.cancel()``. Designed to be cheap to construct, cheap to update,
and trivially extensible (subclass and override ``_refresh`` to add
fields).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from faro.core.run_status import RunHandle, RunStatus

if TYPE_CHECKING:
    from faro.core.controller import Controller


_STATE_COLORS = {
    "pending": "#888888",
    "running": "#2e7d32",
    "cancelling": "#ef6c00",
    "done": "#1565c0",
    "error": "#c62828",
}


class ExperimentStatusWidget(QWidget):
    """A read-out + Stop button for the controller's currently-bound run."""

    def __init__(self, controller: "Controller", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._handle: RunHandle | None = None

        self._build_ui()
        self._refresh(None)

        controller.runStarted.connect(self._on_run_started)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("Experiment status")

        self._state_label = QLabel("idle")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setStyleSheet(
            "font-weight: bold; padding: 4px; border-radius: 4px;"
        )

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate until we have totals
        self._progress.setTextVisible(True)

        form = QFormLayout()
        self._fov_label = QLabel("—")
        self._event_label = QLabel("—")
        self._frames_label = QLabel("0")
        self._lag_label = QLabel("—")
        self._elapsed_label = QLabel("—")
        self._errors_label = QLabel("0")
        form.addRow("Current FOV:", self._fov_label)
        form.addRow("Event index:", self._event_label)
        form.addRow("Frames received:", self._frames_label)
        form.addRow("Lag (ms):", self._lag_label)
        form.addRow("Elapsed:", self._elapsed_label)
        form.addRow("Background errors:", self._errors_label)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._stop_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._state_label)
        layout.addWidget(self._progress)
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addStretch(1)

    # -- run binding --------------------------------------------------------

    def _on_run_started(self, handle: RunHandle) -> None:
        """Re-bind whenever the controller emits ``runStarted``."""
        if self._handle is not None:
            try:
                self._handle.statusChanged.disconnect(self._refresh)
            except Exception:
                pass

        self._handle = handle
        handle.statusChanged.connect(self._refresh)
        self._refresh(handle.status())

    def _on_stop_clicked(self) -> None:
        # cancel() is idempotent and a no-op when the run isn't running;
        # rely on it rather than re-checking is_running here, so the button
        # works even if state is "done"/"error" by the time the click lands.
        if self._handle is not None:
            self._handle.cancel()

    # -- refresh ------------------------------------------------------------

    def _refresh(self, status: RunStatus | None) -> None:
        """Slot connected to ``handle.statusChanged``.

        Also called once with ``None`` at construction time (no handle yet).
        """
        if status is None:
            self._state_label.setText("idle (no run yet)")
            self._state_label.setStyleSheet(
                "font-weight: bold; padding: 4px; border-radius: 4px; "
                f"background-color: {_STATE_COLORS['pending']}; color: white;"
            )
            self._progress.setRange(0, 0)
            self._progress.setValue(0)
            self._progress.setFormat("")
            self._stop_btn.setEnabled(False)
            return

        color = _STATE_COLORS.get(status.state, "#888888")
        self._state_label.setText(status.state.upper())
        self._state_label.setStyleSheet(
            "font-weight: bold; padding: 4px; border-radius: 4px; "
            f"background-color: {color}; color: white;"
        )

        if status.n_events_total > 0:
            self._progress.setRange(0, status.n_events_total)
            self._progress.setValue(status.n_events_consumed)
            self._progress.setFormat(
                f"{status.n_events_consumed} / {status.n_events_total} events"
            )
        else:
            self._progress.setRange(0, 0)
            self._progress.setValue(0)
            self._progress.setFormat("")

        self._fov_label.setText(
            "—" if status.current_fov is None else str(status.current_fov)
        )
        self._event_label.setText(
            "—" if status.current_event_index is None
            else ", ".join(f"{k}={v}" for k, v in status.current_event_index.items())
        )
        self._frames_label.setText(str(status.n_frames_received))
        if status.lag_ms is None:
            self._lag_label.setText("—")
        else:
            self._lag_label.setText(f"{status.lag_ms:+.0f}")

        if status.started_at is not None:
            end = status.finished_at or time.monotonic()
            self._elapsed_label.setText(f"{end - status.started_at:.1f}s")
        else:
            self._elapsed_label.setText("—")

        n_errors = len(status.background_errors)
        if status.fatal_error is not None:
            self._errors_label.setText(
                f"{n_errors} background + fatal: {type(status.fatal_error).__name__}"
            )
        else:
            self._errors_label.setText(str(n_errors))

        # Stop is only meaningful while the run is actually running.
        self._stop_btn.setEnabled(status.state in ("running",))
