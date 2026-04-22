"""Pure-Python fake microscope hardware for tests.

Builds a real :class:`~pymmcore_plus.experimental.unicore.UniMMCore`
instance wired to pure-Python :class:`CameraDevice` and :class:`SLMDevice`
implementations. Tests get the real pymmcore-plus MDA engine, signal
chain, config-group machinery, and ``validate_hardware`` surface
without any C++ device adapters or `.cfg` file on disk.

The :class:`Scene` protocol is the test's plug-in: declare the
microscope's shape (image size, channels, SLM) and provide a
``render(event)`` that maps an :class:`~useq.MDAEvent` to a frame. The
Controller runs through the normal pymmcore-plus path and ``frameReady``
fires with your rendered images.

Usage::

    class MyScene:
        image_height = 256
        image_width = 256
        channels = ["phase-contrast", "stim-405"]
        slm_name = "SLM"
        slm_shape = (256, 256)

        def render(self, event):
            return numpy_image

    core = build_core(MyScene())
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from pymmcore_plus.experimental.unicore import (
    CameraDevice,
    GenericDevice,
    UniMMCore,
)
from pymmcore_plus.experimental.unicore.devices._slm import SLMDevice
from useq import MDAEvent


DEFAULT_CHANNEL_GROUP = "Channel"
CAMERA_LABEL = "Camera"


@runtime_checkable
class Scene(Protocol):
    """Plug-in for :func:`build_core`: image source + hardware shape.

    Required:

    - ``image_height`` / ``image_width``: frame dimensions.
    - ``render(event) -> np.ndarray``: image for each MDAEvent.

    Optional:

    - ``channels``: list of channel-config names to define on the core
      (default ``["phase-contrast"]``). Each config sets ``Camera.Exposure``
      as a no-op property so pymmcore accepts the group.
    - ``channel_group``: name of the channel config group (default
      ``"Channel"``).
    - ``slm_name`` / ``slm_shape``: declare an SLM device; when set,
      :class:`FakeMicroscope` attaches it and stim dispatches flow
      through the real ``setSLMImage`` / ``displaySLMImage`` path.
    - ``on_slm_displayed(mask, event) -> None``: fires after each
      ``displaySLMImage`` with the dispatched mask and the triggering
      event (from ``mmc.mda.events.eventStarted``).
    """

    image_height: int
    image_width: int

    def render(self, event: MDAEvent) -> np.ndarray: ...


class _Bridge:
    """Shared slot for the event currently being executed.

    Pymmcore's camera snap API has no event context; this bridge holds
    the most recent ``eventStarted`` event so snap-time code (camera
    render, SLM record) can attribute its work to the right frame.
    """

    def __init__(self) -> None:
        self.current_event: Optional[MDAEvent] = None

    def _on_event_started(self, event: MDAEvent) -> None:
        self.current_event = event


class FakeCamera(CameraDevice):
    """Camera device that pulls each frame from a :class:`Scene`."""

    def __init__(self, scene: Scene, bridge: _Bridge) -> None:
        super().__init__()
        self._scene = scene
        self._bridge = bridge
        self._exposure = 10.0

    def name(self) -> str:
        return CAMERA_LABEL

    def description(self) -> str:
        return "Pure-Python fake camera for tests"

    def shape(self) -> Tuple[int, int]:
        return (self._scene.image_height, self._scene.image_width)

    def dtype(self):
        return np.uint16

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)

    def get_binning(self) -> int:
        return 1

    def set_binning(self, binning: int) -> None:
        pass

    def start_sequence(self, n: int, get_buffer) -> Iterator[Mapping]:
        for _ in range(n):
            img = self._scene.render(self._bridge.current_event)
            buf = get_buffer(self.shape(), self.dtype())
            buf[:] = img
            yield {"data": buf}


class FakeSLM(SLMDevice):
    """SLM device that records each dispatched mask via the scene hook."""

    def __init__(self, scene: Scene, bridge: _Bridge) -> None:
        super().__init__()
        self._scene = scene
        self._bridge = bridge
        self._buf: Optional[np.ndarray] = None
        self._exposure = 0.0

    def name(self) -> str:
        return self._scene.slm_name

    def description(self) -> str:
        return "Pure-Python fake SLM for tests"

    def shape(self) -> Tuple[int, int]:
        return self._scene.slm_shape

    def dtype(self):
        return np.uint8

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)

    def set_image(self, image: np.ndarray) -> None:
        self._buf = np.asarray(image).copy()

    def get_image(self) -> Optional[np.ndarray]:
        return self._buf

    def display_image(self) -> None:
        on_slm = getattr(self._scene, "on_slm_displayed", None)
        if on_slm is not None and self._buf is not None:
            on_slm(self._buf, self._bridge.current_event)


class _PropertyHolder(GenericDevice):
    """Minimal device that registers arbitrary named properties.

    Used by :func:`build_core` for stubbing devices that
    ``validate_hardware`` needs to see (light sources like ``Spectra``
    with ``*_Level`` properties, etc.). Each property is either a bare
    stringish slot or a float slot with ``(lo, hi)`` limits.
    """

    def __init__(self, name: str, properties: Mapping[str, Optional[Tuple[float, float]]]) -> None:
        super().__init__()
        self._name = name
        self._values: dict[str, object] = {}
        for prop, limits in properties.items():
            default = 0.0 if limits is not None else ""
            self._values[prop] = default
            self.register_property(
                prop,
                getter=lambda self, _p=prop: self._values[_p],
                setter=lambda self, v, _p=prop: self._values.__setitem__(_p, v),
                limits=limits,
            )

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"Fake device: {self._name}"


def build_core(
    scene: Scene,
    *,
    camera_exposure_limits: Optional[Tuple[float, float]] = None,
    extra_devices: Optional[Mapping[str, Mapping[str, Optional[Tuple[float, float]]]]] = None,
    config_data: Optional[Mapping[Tuple[str, str], Iterable[Tuple[str, str, str]]]] = None,
    extra_configs: Optional[Mapping[str, Iterable[str]]] = None,
    channel_group: Optional[str] = None,
) -> UniMMCore:
    """Build a ``UniMMCore`` wired with :class:`FakeCamera` (and SLM).

    Acquisition tests pass a ``Scene`` and nothing else; the returned
    core drives MDA events via ``core.run_mda(events)``. Validation
    tests (``validate_hardware``) pass the extra kwargs to register
    property limits, additional light-source devices, and per-config
    property settings so ``detect_power_properties`` and exposure/power
    range checks find what they expect.

    Args:
        scene: image source + hardware shape declaration.
        camera_exposure_limits: ``(lo, hi)`` to apply to the camera's
            ``Exposure`` property. ``hasPropertyLimits("Camera", "Exposure")``
            will return True and the getters will return these bounds.
        extra_devices: ``{device_name: {property_name: (lo, hi) or None}}``.
            Loads a :class:`_PropertyHolder` per entry so ``validate_hardware``
            sees the device in ``getLoadedDevices()`` with those properties.
        config_data: ``{(group, config_name): [(device, property, value)]}``.
            Defines each config with its settings so ``getConfigData`` walks
            through them. Use this to set up channel configs that reference
            a light source's ``Label`` property (needed for
            ``detect_power_properties`` to match colors).
        extra_configs: ``{group: [config_name, ...]}``. Adds configs
            that appear under ``getAvailableConfigs`` without custom
            settings (each gets a dummy ``Camera.Exposure`` setting).
            Used when a test's config lookup crosses multiple groups.
        channel_group: override ``scene.channel_group`` /
            :data:`DEFAULT_CHANNEL_GROUP`. Passing ``""`` leaves the
            core's channel group unset (some validation tests rely on
            ``getChannelGroup() == ""`` triggering a full config-group
            scan).
    """
    bridge = _Bridge()
    core = UniMMCore()

    camera = FakeCamera(scene, bridge)
    if camera_exposure_limits is not None:
        camera.set_property_limits("Exposure", camera_exposure_limits)
    core.loadPyDevice(CAMERA_LABEL, camera)
    core.initializeDevice(CAMERA_LABEL)
    core.setCameraDevice(CAMERA_LABEL)
    core.setAutoShutter(False)

    slm_name = getattr(scene, "slm_name", None)
    if slm_name:
        slm = FakeSLM(scene, bridge)
        core.loadPyDevice(slm_name, slm)
        core.initializeDevice(slm_name)
        core.setSLMDevice(slm_name)

    # Auto-register any device.property referenced in config_data settings
    # that isn't already in extra_devices. pymmcore's defineConfig requires
    # the device+property to exist; tests shouldn't have to spell it twice.
    merged_devices: dict[str, dict[str, Optional[Tuple[float, float]]]] = {
        d: dict(p) for d, p in (extra_devices or {}).items()
    }
    if config_data:
        for settings in config_data.values():
            for dev, prop, _value in settings:
                if dev == CAMERA_LABEL:
                    continue
                merged_devices.setdefault(dev, {}).setdefault(prop, None)

    for dev_name, props in merged_devices.items():
        holder = _PropertyHolder(dev_name, props)
        core.loadPyDevice(dev_name, holder)
        core.initializeDevice(dev_name)

    configured: set[tuple[str, str]] = set()
    if config_data:
        for (group, config_name), settings in config_data.items():
            for device, prop, value in settings:
                core.defineConfig(group, config_name, device, prop, value)
            configured.add((group, config_name))

    group = (
        channel_group
        if channel_group is not None
        else getattr(scene, "channel_group", DEFAULT_CHANNEL_GROUP)
    )
    default_group = group or DEFAULT_CHANNEL_GROUP
    channels = getattr(scene, "channels", ("phase-contrast",))
    for name in channels:
        if (default_group, name) not in configured:
            core.defineConfig(default_group, name, CAMERA_LABEL, "Exposure", "10")

    if extra_configs:
        for g, names in extra_configs.items():
            for n in names:
                if (g, n) not in configured:
                    core.defineConfig(g, n, CAMERA_LABEL, "Exposure", "10")

    if group:
        core.setChannelGroup(group)

    core.mda.events.eventStarted.connect(bridge._on_event_started)
    return core
