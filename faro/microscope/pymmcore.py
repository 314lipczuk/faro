import atexit
import contextlib
import weakref

from faro.microscope.base import AbstractMicroscope


class PyMMCoreMicroscope(AbstractMicroscope):
    """Intermediate base for all pymmcore-plus-based microscopes.

    Subclasses must set ``self.mmc`` to a ``CMMCorePlus`` instance.

    Implements the :class:`AbstractMicroscope` MDA interface by delegating
    to ``self.mmc`` (run_mda, frameReady signal, cancel, etc.).

    Power properties (mapping channel config -> light-source device/property)
    are auto-detected from the loaded Micro-Manager config.  Subclasses may
    set ``POWER_PROPERTIES`` to override or supplement the auto-detected
    values.

    On construction, an atexit hook is registered that cancels any running
    MDA and unloads every Micro-Manager device when the interpreter shuts
    down. Without this, device adapters can stay bound to the dying Python
    process and prevent the next process from acquiring the same
    configuration. Subclasses with extra teardown (background threads,
    serial ports, etc.) should override :meth:`_teardown_hardware` rather
    than re-register their own hooks.
    """

    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    POWER_PROPERTIES: dict[str, tuple[str, str]] = {}

    def __init__(self):
        super().__init__()
        self.mmc = None  # subclasses must set this
        self._detected_power_properties: dict[str, tuple[str, str]] | None = None
        self._current_group: str | None = None

        # Register cleanup via a weakref so the hook doesn't pin the
        # microscope instance. self.mmc is checked at fire time because
        # subclasses set it after super().__init__() returns.
        weak_self = weakref.ref(self)

        def _atexit_teardown() -> None:
            scope = weak_self()
            if scope is None:
                return
            scope._teardown_hardware()

        atexit.register(_atexit_teardown)
        self._atexit_teardown = _atexit_teardown

    def _teardown_hardware(self) -> None:
        """Release all hardware held by this microscope.

        Called from the atexit hook registered in :meth:`__init__`, and
        also reusable as an explicit teardown step from subclasses that
        expose a public ``shutdown`` API. Override in subclasses to add
        extra cleanup (stopping background threads, closing serial ports,
        etc.) — call ``super()._teardown_hardware()`` last so device
        unload happens after subclass-owned threads have stopped.

        Suppresses exceptions so a flaky device can't prevent the rest
        of the teardown (or, in the atexit path, the interpreter's
        finalization) from running.
        """
        if self.mmc is None:
            return
        with contextlib.suppress(Exception):
            self.mmc.mda.cancel()
        with contextlib.suppress(Exception):
            self.mmc.unloadAllDevices()

    # ------------------------------------------------------------------
    # MDA interface implementation
    # ------------------------------------------------------------------

    def run_mda(self, event_iter):
        return self.mmc.run_mda(event_iter)

    def connect_frame(self, callback):
        self.mmc.mda.events.frameReady.connect(callback)

    def disconnect_frame(self, callback):
        self.mmc.mda.events.frameReady.disconnect(callback)

    def cancel_mda(self):
        self.mmc.mda.cancel()

    def resolve_group(self, config_name: str) -> str:
        """Return the channel group for *config_name*, auto-detecting if needed."""
        if self._current_group is None:
            self._current_group = self.mmc.getChannelGroup()
        if self._current_group:
            return self._current_group
        # getChannelGroup() was empty — find a group containing this preset
        for group in self.mmc.getAvailableConfigGroups():
            if config_name in self.mmc.getAvailableConfigs(group):
                self._current_group = group
                return group
        return ""

    def resolve_power(self, channel):
        """Return (device, property, power) for a PowerChannel, or None."""
        power = getattr(channel, "power", None)
        if power is None:
            return None
        mapping = self.get_power_properties().get(channel.config)
        if mapping is None:
            return None
        device_name, property_name = mapping
        return (device_name, property_name, power)

    # ------------------------------------------------------------------
    # Power property management
    # ------------------------------------------------------------------

    def detect_power_properties(self, group=None) -> dict[str, tuple[str, str]]:
        """Auto-detect power properties from the loaded Micro-Manager config.

        Scans for devices with ``*_Level`` properties (e.g. Spectra, LedDMD)
        and matches channel config presets to their LED color.

        Call this after ``mmc.loadSystemConfiguration()`` to populate the
        mapping.  Results are cached; call with ``group`` to restrict the scan.
        Manual ``POWER_PROPERTIES`` always take priority over auto-detected ones.
        """
        if self.mmc is None:
            return {}
        from faro.core.utils import detect_power_properties
        detected = detect_power_properties(self.mmc, group=group)
        self._detected_power_properties = detected
        return detected

    def get_power_properties(self) -> dict[str, tuple[str, str]]:
        """Return merged power properties (auto-detected + manual overrides)."""
        detected = self._detected_power_properties or {}
        # Manual POWER_PROPERTIES override auto-detected ones
        return {**detected, **self.POWER_PROPERTIES}

    def validate_hardware(self, events) -> bool:
        # Materialize once so both the base and util checks can iterate.
        events = list(events)
        ok = super().validate_hardware(events)
        if self.mmc is None:
            return ok  # nothing else to validate against
        # Auto-detect on first use if not yet done
        if self._detected_power_properties is None:
            self.detect_power_properties()
        from faro.core.utils import validate_hardware
        ok_mmc = validate_hardware(
            events, self.mmc, power_properties=self.get_power_properties()
        )
        return ok and ok_mmc
