"""Qt status widget for the controller's currently-bound run.

Construct with a :class:`faro.core.controller.Controller` instance::

    from faro.widgets import ExperimentStatusWidget
    widget = ExperimentStatusWidget(ctrl)
    viewer.window.add_dock_widget(widget, name="Experiment")

Layout (top to bottom):

  - State label  (RUNNING / DONE / CANCELLING / ERROR, color-coded)
  - Legend chips (imaging / stim / ref) -- the chip matching the *current*
    event type is fully opaque; the others are dimmed
  - Event strip  (one cell per RTMEvent, color-coded by type; past cells
    are fully opaque, future cells are dimmed; the current cell has a
    darker border)
  - FOV map      (one dot per unique FOV position, equal-aspect, with a
    grey path drawn in visit order; the dot for the current FOV is
    re-colored in the active event-type's color)
  - Stats form   (event N/M, elapsed, scheduled, lag, remaining, frames,
    background errors)
  - Stop button  (calls ``handle.cancel()``)

The widget subscribes to ``ctrl.runStarted``, so it automatically re-binds
to whichever run is current. Each ``RunHandle.statusChanged`` emission
updates the labels / strip / map; a small QTimer also refreshes the
``elapsed`` / ``remaining`` fields between status updates so the clock
doesn't appear frozen between frames.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

from qtpy.QtCore import Qt, QTimer, QPointF, QRectF
from qtpy.QtGui import (
    QBrush, QColor, QFontDatabase, QPainter, QPalette, QPen,
)
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from faro.core.data_structures import ImgType
from faro.core.run_status import RunHandle, RunStatus

if TYPE_CHECKING:
    from faro.core.controller import Controller


# ─────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────

EVENT_COLORS: dict[str, str] = {
    "imaging": "#2e7d32",
    "stim":    "#1565c0",
    "ref":     "#ef6c00",
}
DEFAULT_EVENT_COLOR = "#888888"

# Corner radius -- matches napari's own widgets (buttons, layer controls).
_RADIUS_PX = 3

# Translucent neutral overlay for the grouped panels (FOV map / stats).
# A 50%-grey at low alpha lightens a dark theme and darkens a light one,
# so the grouping reads on both without hardcoding a theme color.
_PANEL_BG = "rgba(128, 128, 128, 28)"

# Event strip
_FUTURE_ALPHA      = 90
_PAST_ALPHA        = 255
_BORDER_PX         = 2
_GAP_PX            = 1
_MIN_GAP_AT_CELL_W = 3.0

# FOV map
_DOT_RADIUS_PX     = 5
_PATH_WIDTH_PX     = 2
_MAP_PADDING_PX    = 24
_MIN_WORLD_EXTENT  = 1e-6

# Lag warn threshold (red is recognizable on both light and dark themes)
_LAG_WARN_S        = 5.0
_LAG_BAD_COLOR     = "#e53935"


# ─────────────────────────────────────────────────────────────────────────
# Helpers (event-list introspection + small formatters)
# ─────────────────────────────────────────────────────────────────────────

def _event_type_token(ev) -> str:
    """Map an RTMEvent to one of {"ref", "stim", "imaging"} for visualisation.

    Order of precedence matches what the user sees as the *dominant* effect
    for that timepoint: ref > stim > imaging.

    RTMEvent doesn't expose ``stim`` / ``ref`` booleans directly --
    ``events_to_dataframe`` derives them from the channel tuple lengths,
    so we do the same here.
    """
    if getattr(ev, "ref_channels", ()):
        return "ref"
    if getattr(ev, "stim_channels", ()):
        return "stim"
    # Fallback for plain MDAEvents: peek at metadata['img_type']
    md = getattr(ev, "metadata", None) or {}
    img_type = md.get("img_type")
    if img_type == ImgType.IMG_REF:
        return "ref"
    if img_type == ImgType.IMG_STIM:
        return "stim"
    return "imaging"


def _extract_plan(events: Sequence) -> tuple[list[str], list[int], list[tuple[float, float]], list[float]]:
    """Walk events once and return (types, fovs, positions_by_fov, scheduled).

    - ``types``: per-event type token
    - ``fovs``:  per-event FOV index
    - ``positions_by_fov``: list indexed by FOV index, holding ``(x, y)``
      of that FOV's stage position (taken from the first event that visits
      each FOV).
    - ``scheduled``: per-event ``min_start_time`` in seconds (0.0 if missing)
    """
    types: list[str] = []
    fovs: list[int] = []
    scheduled: list[float] = []
    seen_fov: dict[int, tuple[float, float]] = {}

    for ev in events:
        types.append(_event_type_token(ev))
        p = ev.index.get("p", 0)
        fovs.append(p)
        if p not in seen_fov:
            seen_fov[p] = (float(ev.x_pos or 0.0), float(ev.y_pos or 0.0))
        mst = getattr(ev, "min_start_time", None)
        scheduled.append(float(mst) if mst is not None else 0.0)

    if seen_fov:
        max_p = max(seen_fov)
        positions = [seen_fov.get(i, (0.0, 0.0)) for i in range(max_p + 1)]
    else:
        positions = []
    return types, fovs, positions, scheduled


def format_duration(seconds: float, *, show_ms: bool = False) -> str:
    """``hh:mm:ss h`` / ``mm:ss min`` / ``s s`` with optional .mmm.

    Leading components are dropped when zero, and the largest displayed
    component is suffixed with its unit ("h" / "min" / "s") so the unit
    stays explicit even after truncation.
    """
    if seconds < 0:
        return "-" + format_duration(-seconds, show_ms=show_ms)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if show_ms:
        if h:
            return f"{h:d}:{m:02d}:{s:06.3f} h"
        if m:
            return f"{m:d}:{s:06.3f} min"
        return f"{s:.3f} s"
    s_int = int(s)
    if h:
        return f"{h:d}:{m:02d}:{s_int:02d} h"
    if m:
        return f"{m:d}:{s_int:02d} min"
    return f"{s_int:d} s"


def _runs(seq: Sequence[str]):
    """Yield (start, end_exclusive, value) for each contiguous run."""
    if not seq:
        return
    start, cur = 0, seq[0]
    for i in range(1, len(seq)):
        if seq[i] != cur:
            yield start, i, cur
            start, cur = i, seq[i]
    yield start, len(seq), cur


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _chip_style(color_hex: str, *, active: bool) -> str:
    r, g, b = _hex_to_rgb(color_hex)
    if active:
        return (
            f"background-color: rgba({r},{g},{b},255); color: white; "
            f"padding: 2px 8px; border-radius: {_RADIUS_PX}px; font-weight: bold;"
        )
    return (
        f"background-color: rgba({r},{g},{b},60); color: rgba({r},{g},{b},180); "
        f"padding: 2px 8px; border-radius: {_RADIUS_PX}px; font-weight: bold;"
    )


# ─────────────────────────────────────────────────────────────────────────
# EventStrip
# ─────────────────────────────────────────────────────────────────────────

class EventStrip(QWidget):
    """Horizontal strip with one cell per event, color-coded by type.

    Past + current cells are fully opaque (acts as a progress bar). Future
    cells are dimmed. Same-type contiguous runs are merged into a single
    fill so 1000s of events still render correctly with consistent alpha
    rather than the over-stacking that per-cell alpha causes at sub-pixel
    cell widths.
    """

    def __init__(self, event_types: Sequence[str], parent: QWidget | None = None):
        super().__init__(parent)
        self._types = list(event_types)
        self._current = -1
        self.setMinimumHeight(20)
        self.setMinimumWidth(120)

    def set_types(self, event_types: Sequence[str]) -> None:
        self._types = list(event_types)
        self._current = -1
        self.update()

    def set_current(self, index: int) -> None:
        if index != self._current:
            self._current = index
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)

        # Empty state: grey rounded placeholder so the timeline region is
        # visible before a run is loaded (replaced by the colored progress
        # bar once events arrive).
        if not self._types:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(128, 128, 128, 28)))
            painter.drawRoundedRect(QRectF(self.rect()), _RADIUS_PX, _RADIUS_PX)
            placeholder = self.palette().color(QPalette.ColorRole.WindowText)
            placeholder.setAlpha(128)
            painter.setPen(QPen(placeholder))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "(no events loaded)",
            )
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        n = len(self._types)
        w = self.width()
        h = self.height()
        gap = _GAP_PX if (w / max(n, 1)) >= _MIN_GAP_AT_CELL_W else 0
        cell_w = (w - max(0, n - 1) * gap) / n
        stride = cell_w + gap

        def x_for(i: int) -> float:
            return i * stride

        def width_for(span: int) -> float:
            return max(cell_w * span + max(0, span - 1) * gap, 1.0)

        # Future / dim layer
        for start, end, t in _runs(self._types):
            color = QColor(EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR))
            color.setAlpha(_FUTURE_ALPHA)
            painter.fillRect(QRectF(x_for(start), 0, width_for(end - start), h), color)

        # Past + current overlay
        if self._current >= 0:
            past_end = min(self._current + 1, n)
            for start, end, t in _runs(self._types[:past_end]):
                color = QColor(EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR))
                color.setAlpha(_PAST_ALPHA)
                painter.fillRect(
                    QRectF(x_for(start), 0, width_for(end - start), h), color
                )

        # Active border
        if 0 <= self._current < n:
            t = self._types[self._current]
            pen = QPen(QColor(EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR)).darker(160))
            pen.setWidth(_BORDER_PX)
            painter.setPen(pen)
            visible_w = max(cell_w, 3.0)
            x = self._current * stride
            x = min(max(0.0, x - (visible_w - cell_w) / 2), w - visible_w)
            painter.drawRect(QRectF(x + 0.5, 0.5, visible_w - 1, h - 1))


# ─────────────────────────────────────────────────────────────────────────
# FovMap
# ─────────────────────────────────────────────────────────────────────────

class FovMap(QWidget):
    """Equal-aspect map of FOV positions with a visit-order path.

    The active dot is recolored by ``set_current(idx, color=...)`` so it
    can match the current event type's color. Background is transparent so
    napari's theme shows through; the drawing area is constrained to the
    largest centered square so the widget never appears taller than wide.
    """

    def __init__(
        self,
        positions: Sequence[tuple[float, float]],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._positions = list(positions)
        self._current = -1
        self._active_color = EVENT_COLORS["imaging"]
        self.setMinimumWidth(160)
        # Expand to the available width; height is pinned square in
        # resizeEvent. heightForWidth is intentionally NOT used --
        # QVBoxLayout honors it poorly and ends up distributing slack
        # space above and below the widget.
        from qtpy.QtWidgets import QSizePolicy
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setFixedHeight(160)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep the widget square: height tracks width. setFixedHeight to
        # the same value is a no-op, so this settles after one relayout.
        if self.height() != self.width():
            self.setFixedHeight(self.width())

    def set_positions(self, positions: Sequence[tuple[float, float]]) -> None:
        self._positions = list(positions)
        self._current = -1
        self.update()

    def set_current(self, index: int, color: str | None = None) -> None:
        changed = (
            index != self._current
            or (color is not None and color != self._active_color)
        )
        if color is not None:
            self._active_color = color
        self._current = index
        if changed:
            self.update()

    def _world_to_screen(self) -> tuple[float, float, float]:
        if not self._positions:
            return 1.0, 0.0, 0.0
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(x_max - x_min, _MIN_WORLD_EXTENT)
        y_span = max(y_max - y_min, _MIN_WORLD_EXTENT)
        # Confine the drawing to the largest centered square -- so even if
        # the widget ends up wider than tall (or vice-versa), the dots stay
        # together in a square region rather than getting spread thin
        # along the longer axis.
        side = min(self.width(), self.height())
        usable = max(1.0, side - 2 * _MAP_PADDING_PX)
        scale_x = usable / x_span if x_span > _MIN_WORLD_EXTENT * 10 else float("inf")
        scale_y = usable / y_span if y_span > _MIN_WORLD_EXTENT * 10 else float("inf")
        scale = min(scale_x, scale_y)
        if scale == float("inf"):
            scale = 1.0
        x_off = self.width() / 2 - scale * (x_min + x_max) / 2
        y_off = self.height() / 2 + scale * (y_min + y_max) / 2  # +y up
        return scale, x_off, y_off

    def _to_screen(self, x: float, y: float) -> QPointF:
        s, xo, yo = self._world_to_screen()
        return QPointF(s * x + xo, yo - s * y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Panel background -- the map paints its own rounded-rect fill (same
        # translucent neutral as the stats panel) rather than sitting in a
        # QFrame, so the FOV-counter text lands exactly at the panel's
        # top-left corner regardless of how the layout sizes the widget.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(128, 128, 128, 28)))
        painter.drawRoundedRect(QRectF(self.rect()), _RADIUS_PX, _RADIUS_PX)

        # FOV counter, pinned to the panel's top-left corner.
        n = len(self._positions)
        if n == 0:
            label = "FOV -/-"
        else:
            cur_txt = str(self._current + 1) if 0 <= self._current < n else "-"
            label = f"FOV {cur_txt}/{n}"
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        painter.setPen(QPen(text_color))
        painter.drawText(
            QPointF(8, 6 + painter.fontMetrics().ascent()), label
        )

        if not self._positions:
            return

        # Inactive dots / path use napari's foreground text color at 50%
        # alpha, so the map reads correctly on both light and dark themes.
        inactive = self.palette().color(QPalette.ColorRole.WindowText)
        inactive.setAlpha(128)

        pts = [self._to_screen(*p) for p in self._positions]

        if len(pts) >= 2:
            pen = QPen(inactive)
            pen.setWidth(_PATH_WIDTH_PX)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for a, b in zip(pts[:-1], pts[1:]):
                painter.drawLine(a, b)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(inactive))
        for pt in pts:
            painter.drawEllipse(pt, _DOT_RADIUS_PX, _DOT_RADIUS_PX)

        if 0 <= self._current < len(pts):
            painter.setBrush(QBrush(QColor(self._active_color)))
            painter.drawEllipse(pts[self._current], _DOT_RADIUS_PX, _DOT_RADIUS_PX)


# ─────────────────────────────────────────────────────────────────────────
# ExperimentStatusWidget
# ─────────────────────────────────────────────────────────────────────────

class ExperimentStatusWidget(QWidget):
    """Read-out + Stop button for the controller's currently-bound run."""

    def __init__(self, controller: "Controller", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._handle: RunHandle | None = None

        # Cached plan derived from handle.events at run start
        self._event_types: list[str] = []
        self._event_fovs: list[int] = []
        self._scheduled: list[float] = []
        self._total_duration: float = 0.0

        self._build_ui()
        self._refresh(None)

        controller.runStarted.connect(self._on_run_started)

        # Drain psygnal's queued callbacks via a main-thread QTimer; without
        # this, statusChanged emissions from the worker thread would sit in
        # psygnal's _GLOBAL_QUEUE forever. Idempotent across widgets.
        try:
            from psygnal.qt import start_emitting_from_queue
            start_emitting_from_queue()
        except ImportError:
            pass

        # Tick the elapsed/remaining clocks between statusChanged emissions
        # so the time fields don't visibly freeze between frames.
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(250)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("Experiment status")

        # ── State chip -- translucent neutral fill (matches the FOV map /
        # stats panels), so it keeps the legend-chip shape without
        # competing with the imaging/stim/ref colors.
        self._state_label = QLabel("idle")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; "
            f"border-radius: {_RADIUS_PX}px; "
            f"background-color: {_PANEL_BG};"
        )

        # ── Legend chips
        self._legend_chips: dict[str, QLabel] = {}
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(6)
        for label, key in [("imaging", "imaging"), ("stim", "stim"), ("ref", "ref")]:
            chip = QLabel(label)
            chip.setStyleSheet(_chip_style(EVENT_COLORS[key], active=False))
            self._legend_chips[key] = chip
            legend_row.addWidget(chip)
        legend_row.addStretch(1)

        # ── Strip + map
        self._strip = EventStrip([])
        self._map = FovMap([])

        # ── Stats form -- inherit napari's font/palette; only the time
        # values get the platform's fixed-width font (column alignment).
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._event_value     = QLabel("-/-")
        self._elapsed_value   = QLabel("-")
        self._scheduled_value = QLabel("-")
        self._lag_value       = QLabel("-")
        self._remaining_value = QLabel("-")
        self._errors_value    = QLabel("-")
        right_align = (
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        for w in (
            self._event_value, self._elapsed_value, self._scheduled_value,
            self._lag_value, self._remaining_value, self._errors_value,
        ):
            w.setAlignment(right_align)
        for w in (self._elapsed_value, self._scheduled_value,
                  self._lag_value, self._remaining_value):
            w.setFont(mono)

        form = QFormLayout()
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(2)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        # Stretch the value column so right-aligned text lands at the
        # widget's right edge instead of hugging the label.
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.addRow("event:",     self._event_value)
        form.addRow("elapsed:",   self._elapsed_value)
        form.addRow("scheduled:", self._scheduled_value)
        form.addRow("lag:",       self._lag_value)
        form.addRow("remaining:", self._remaining_value)
        form.addRow("errors:",    self._errors_value)

        # ── Stats panel: a subtly-shaded frame echoing napari's boxed
        # layer-controls sections. (The FOV map paints its own matching
        # background in paintEvent, so it isn't wrapped in a frame.)
        stats_panel = QFrame()
        stats_panel.setObjectName("faroPanel")
        stats_panel_layout = QVBoxLayout(stats_panel)
        stats_panel_layout.setContentsMargins(0, 0, 0, 0)
        stats_panel_layout.addLayout(form)
        stats_panel.setStyleSheet(
            f"QFrame#faroPanel {{ background-color: {_PANEL_BG}; "
            f"border-radius: {_RADIUS_PX}px; }}"
        )

        # ── Pause + Stop buttons
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._pause_btn.setEnabled(False)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._pause_btn)
        button_row.addWidget(self._stop_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._state_label)
        layout.addLayout(legend_row)
        layout.addWidget(self._strip)
        # No stretch on the map -- resizeEvent pins it square, and the
        # trailing stretch absorbs leftover vertical space below.
        layout.addWidget(self._map)
        layout.addWidget(stats_panel)
        layout.addLayout(button_row)
        layout.addStretch(1)

    # -- run binding --------------------------------------------------------

    def _on_run_started(self, handle: RunHandle) -> None:
        """Re-bind to a new run; rebuild the strip + map from its events."""
        if self._handle is not None:
            try:
                self._handle.statusChanged.disconnect(self._refresh)
            except Exception:
                pass

        self._handle = handle
        # thread="main" routes worker-thread emits through psygnal's main-
        # thread queue (drained by start_emitting_from_queue's QTimer in
        # __init__). Without it the slot would run synchronously off the
        # worker thread and touch QWidgets / OpenGL from the wrong thread.
        handle.statusChanged.connect(self._refresh, thread="main")

        # Rebuild plan-derived widgets from the events list, if we have one
        events = getattr(handle, "events", None) or []
        types, fovs, positions, scheduled = _extract_plan(events)
        self._event_types = types
        self._event_fovs = fovs
        self._scheduled = scheduled
        self._total_duration = scheduled[-1] if scheduled else 0.0
        self._strip.set_types(types)
        self._map.set_positions(positions)

        self._refresh(handle.status())

    def _on_stop_clicked(self) -> None:
        if self._handle is not None:
            self._handle.cancel()

    def _on_pause_clicked(self) -> None:
        """Toggle pause / resume on the bound handle."""
        if self._handle is None:
            return
        if self._handle.is_paused():
            self._handle.resume()
        else:
            self._handle.pause()

    # -- refresh ------------------------------------------------------------

    def _refresh(self, status: RunStatus | None) -> None:
        """Slot connected to ``handle.statusChanged``.

        Also called once with ``None`` at construction time (no handle yet).
        """
        if status is None:
            self._render_idle()
            return

        # ── State banner -- plain bold text, no background fill (a colored
        # banner clashed with the imaging/stim/ref legend colors).
        self._state_label.setText(status.state.upper())

        # ── Strip + map cursor
        cur_idx = self._current_index(status)
        n_total = status.n_events_total or len(self._event_types)

        if 0 <= cur_idx < len(self._event_types):
            t = self._event_types[cur_idx]
            color = EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR)
            self._strip.set_current(cur_idx)
            fov_for_event = self._event_fovs[cur_idx]
            self._map.set_current(fov_for_event, color=color)
            self._update_legend(active_type=t)
        else:
            self._strip.set_current(-1)
            self._map.set_current(-1)
            self._update_legend(active_type=None)

        # ── Stats: event index
        self._event_value.setText(
            f"{cur_idx + 1} / {n_total}" if n_total else "-/-"
        )

        # ── Stats: elapsed, scheduled, lag, remaining
        self._render_time_fields(status, cur_idx)

        # ── Errors
        n_errors = len(status.background_errors)
        if status.fatal_error is not None:
            self._errors_value.setText(
                f"{n_errors} bg + fatal: {type(status.fatal_error).__name__}"
            )
            self._errors_value.setStyleSheet(
                f"color: {_LAG_BAD_COLOR}; font-weight: bold;"
            )
        else:
            self._errors_value.setText(str(n_errors))
            self._errors_value.setStyleSheet("")

        # ── Buttons
        self._update_buttons(status.state)

    def _update_buttons(self, state: str) -> None:
        """Enable/label Pause + Stop according to the run state."""
        # Pause/Resume is meaningful only while the run is live.
        live = state in ("running", "pausing", "paused")
        self._pause_btn.setEnabled(live)
        if state in ("paused", "pausing"):
            self._pause_btn.setText("Resume")
        else:
            self._pause_btn.setText("Pause")
        # Stop is meaningful while running OR paused (cancel breaks the
        # pause-wait too).
        self._stop_btn.setEnabled(live)

    def _render_idle(self) -> None:
        self._state_label.setText("idle (no run yet)")
        self._strip.set_current(-1)
        self._map.set_current(-1)
        self._update_legend(active_type=None)
        self._event_value.setText("-/-")
        for w in (self._elapsed_value, self._scheduled_value,
                  self._lag_value, self._remaining_value):
            w.setText("-")
        self._lag_value.setStyleSheet("")
        self._errors_value.setText("-")
        self._errors_value.setStyleSheet("")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")
        self._stop_btn.setEnabled(False)

    @staticmethod
    def _current_index(status: RunStatus) -> int:
        """Index of the most-recently-*snapped* event (-1 if none yet).

        Uses ``n_frames_received`` (actual imaging/ref snaps) rather than
        ``n_events_consumed`` (the feed loop's queue-ahead position). The
        feed loop runs 3-4 events ahead of the engine because of the
        backpressure window, so keying off it would make the strip jump
        and the scheduled-time field flicker between the queued event and
        the snapped one. Both ``_refresh`` and the ``_tick`` QTimer call
        this so they always agree on "current".
        """
        return status.n_frames_received - 1 if status.n_frames_received > 0 else -1

    def _render_time_fields(self, status: RunStatus, cur_idx: int) -> None:
        # Elapsed: prefer (now - started_at); fall back to last_frame_wallclock.
        if status.started_at is not None:
            if status.finished_at is not None:
                elapsed = status.finished_at - status.started_at
            else:
                elapsed = time.monotonic() - status.started_at
        else:
            elapsed = None

        scheduled = (
            self._scheduled[cur_idx]
            if 0 <= cur_idx < len(self._scheduled)
            else None
        )

        # lag: prefer the controller's per-frame measurement; fall back to
        # elapsed-vs-scheduled if the per-frame value isn't populated yet.
        lag_s: float | None = None
        if status.lag_ms is not None:
            lag_s = status.lag_ms / 1000.0
        elif elapsed is not None and scheduled is not None:
            lag_s = elapsed - scheduled

        remaining: float | None = None
        if elapsed is not None and self._total_duration:
            remaining = max(0.0, self._total_duration - elapsed)

        self._elapsed_value.setText(
            format_duration(elapsed) if elapsed is not None else "-"
        )
        self._scheduled_value.setText(
            format_duration(scheduled) if scheduled is not None else "-"
        )
        if lag_s is None:
            self._lag_value.setText("-")
            self._lag_value.setStyleSheet("")
        else:
            sign = "+" if lag_s >= 0 else ""
            self._lag_value.setText(f"{sign}{format_duration(lag_s, show_ms=True)}")
            if lag_s > _LAG_WARN_S:
                self._lag_value.setStyleSheet(
                    f"color: {_LAG_BAD_COLOR}; font-weight: bold;"
                )
            else:
                self._lag_value.setStyleSheet("")
        self._remaining_value.setText(
            format_duration(remaining) if remaining is not None else "-"
        )

    def _update_legend(self, active_type: str | None) -> None:
        for key, chip in self._legend_chips.items():
            chip.setStyleSheet(
                _chip_style(EVENT_COLORS[key], active=(key == active_type))
            )

    def _tick(self) -> None:
        """QTimer slot: refresh time-derived fields between status emissions."""
        if self._handle is None:
            return
        status = self._handle.status()
        # Keep the clock live while the run is active -- including while
        # paused, since wall-clock elapsed (and thus lag) keeps growing.
        if status.state not in ("running", "pausing", "paused"):
            return
        self._render_time_fields(status, self._current_index(status))
