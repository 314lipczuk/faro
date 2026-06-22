"""Pure-Python unit tests for Pertzlab-specific faro code.

Lives under ``tests/hardware/pertzlab/`` because its subjects are
Pertzlab-scope-only: per-microscope power-property mappings (declared
manually; an unmapped ``PowerChannel`` fails loud rather than silently
dropping the requested power) and
:class:`faro.microscope.pertzlab.moench.MoenchMDAEngine`'s
``SKIP_WAIT_DEVICES`` filter.

These do **not** require a real scope and are **not** marked
``@pytest.mark.hardware``; they run in every test session.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from faro.core.data_structures import Channel, PowerChannel


# ===================================================================
# Power-property mapping (manual-only; unmapped power fails loud)
# ===================================================================

class TestPowerPropertyMapping:
    """Power mappings are declared on the microscope; no auto-detection."""

    def _mic(self, mapping):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        mic.POWER_PROPERTIES = mapping
        return mic

    def test_get_power_properties_is_manual_only(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.get_power_properties() == {"CyanStim": ("LED", "Cyan_Level")}

    def test_resolve_power_mapped(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="CyanStim", exposure=10, power=25)
        assert mic.resolve_power(ch) == ("LED", "Cyan_Level", 25)

    def test_resolve_power_none_without_power(self):
        """Plain Channels and power-less PowerChannels resolve to None."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.resolve_power(Channel(config="BF", exposure=10)) is None
        assert mic.resolve_power(PowerChannel(config="x", exposure=10)) is None

    def test_resolve_power_raises_when_unmapped(self):
        """A PowerChannel with power set but no mapping must fail loud."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="mScarlet3", exposure=10, power=10)
        with pytest.raises(ValueError, match="mScarlet3"):
            mic.resolve_power(ch)

    def test_validate_hardware_flags_unmapped_power(self):
        """validate_hardware warns + fails when a power channel has no mapping."""
        from faro.core.utils import validate_hardware

        class _StubMMC:
            def getAvailableConfigGroups(self):
                return ["TTL_ERK"]

            def getAvailableConfigs(self, group):
                return ["mScarlet3"]

            def getCameraDevice(self):
                return ""  # skip the exposure-limit block

        events = [
            SimpleNamespace(
                channels=[PowerChannel(config="mScarlet3", exposure=200, power=10)],
                stim_channels=[],
                ref_channels=[],
            )
        ]
        with pytest.warns(UserWarning, match="no power-property mapping"):
            ok = validate_hardware(events, _StubMMC(), power_properties={})
        assert ok is False


# ===================================================================
# SKIP_WAIT_DEVICES — per-microscope waitForDevice skip list
# ===================================================================


class _FakeWaitMMCore:
    """Minimal mmcore surface for _wait_for_system_excluding_xy tests."""

    def __init__(self, devices: list[str], xy_stage: str = ""):
        self._devices = devices
        self._xy_stage = xy_stage
        self.wait_calls: list[str] = []

    def getLoadedDevices(self):
        return self._devices

    def getXYStageDevice(self):
        return self._xy_stage

    def waitForDevice(self, dev: str):
        self.wait_calls.append(dev)

    def getXYPosition(self):
        return (0.0, 0.0)


class _FakeMoench:
    """Weakref-able stand-in for Moench carrying SKIP_WAIT_DEVICES."""

    def __init__(self, skip: tuple[str, ...]):
        self.SKIP_WAIT_DEVICES = skip


class TestSkipWaitDevices:
    """MoenchMDAEngine honors the microscope's SKIP_WAIT_DEVICES tuple.

    Regression guard against the 5 s-per-event Mosaic3 stuck-Busy wait
    (TODO.md #1): devices listed here must never hit waitForDevice().
    """

    def _make_engine(self, mmc, mic):
        import weakref

        from faro.microscope.pertzlab.moench import MoenchMDAEngine

        # Bypass MDAEngine.__init__ — the base class wants a real
        # CMMCorePlus, but this test only exercises the skip-filter
        # logic which reads self.mmcore and self.microscope.
        engine = MoenchMDAEngine.__new__(MoenchMDAEngine)
        engine._mmcore_ref = weakref.ref(mmc)
        engine._microscope_ref = weakref.ref(mic)
        return engine

    def test_skip_devices_bypass_wait(self):
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Shutter", "Mosaic3", "XYStage"],
            xy_stage="XYStage",
        )
        mic = _FakeMoench(skip=("Mosaic3",))
        engine = self._make_engine(mmc, mic)

        engine._wait_for_system_excluding_xy(MDAEvent())

        assert "Mosaic3" not in mmc.wait_calls, "Mosaic3 should be skipped"
        assert "XYStage" not in mmc.wait_calls, "xy_stage handled separately"
        assert "Core" not in mmc.wait_calls, "Core is always skipped"
        assert "Camera" in mmc.wait_calls
        assert "Shutter" in mmc.wait_calls

    def test_missing_attribute_is_noop(self):
        """Microscopes without SKIP_WAIT_DEVICES fall through to the default."""
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Mosaic3"],
            xy_stage="",
        )

        class _BareMic:
            pass

        engine = self._make_engine(mmc, _BareMic())
        engine._wait_for_system_excluding_xy(MDAEvent())

        # Without SKIP_WAIT_DEVICES, Mosaic3 is waited on as before.
        assert "Mosaic3" in mmc.wait_calls
        assert "Camera" in mmc.wait_calls
