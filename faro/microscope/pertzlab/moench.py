import pymmcore_plus
import weakref

from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.core.data_structures import ImgType, PowerChannel
from faro.core.dmd import DMD
from faro.core._useq_compat import SLMImage
from pymmcore_plus.mda._engine import MDAEngine
from typing import Optional
from pymmcore_plus._logger import logger

from useq import MDAEvent
import os
import time
import threading
import locale
import logging
from pymmcore_plus.core._sequencing import SequencedEvent, iter_sequenced_events
from contextlib import suppress


os.environ["PYMM_PARALLEL_INIT"] = "0"


def _set_c_numeric_locale():
    """Set locale to C/POSIX to ensure period as decimal separator."""
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
    except locale.Error:
        for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                break
            except locale.Error:
                continue


def _pump_qt_events() -> None:
    """Process pending Qt events if a Qt app is running; no-op otherwise.

    Used by ``calibrate_dmd(background=True)`` to keep napari responsive
    while the calibration MDAs run on a worker thread.
    """
    try:
        from qtpy.QtCore import QCoreApplication
    except Exception:
        return
    app = QCoreApplication.instance()
    if app is not None:
        app.processEvents()


class KeepDMDAlive:
    def __init__(self, mmc, dmd):
        self.mmc = mmc
        self.dmd = dmd
        self.thread: threading.Thread | None = None
        self.last_wakeup = 0.0
        # daemon=True so interpreter shutdown doesn't block on this
        # thread holding COM3 (zombie python.exe on next session).
        self._stop_event = threading.Event()

    def wakeup_dmd(self):
        # Re-display the DMD's current live-view pattern (all-on by default,
        # or e.g. a checkerboard set for a focus check) so it survives the
        # periodic refresh instead of being forced back to all-on.
        self.dmd.display_livemode()

    def run(self):
        _set_c_numeric_locale()
        self._stop_event.clear()
        self.last_wakeup = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            current_time = time.time()
            if current_time - self.last_wakeup > 60:  # Wake up every minute
                self.wakeup_dmd()
                self.last_wakeup = current_time
            # Event.wait lets stop() break out immediately instead of
            # eating up to 5 s of teardown time per session.
            if self._stop_event.wait(timeout=5):
                return

    def stop(self):
        _set_c_numeric_locale()
        self._stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = None
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 100)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())


class Moench(PyMMCoreMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0_api75"
    MICROMANAGER_CONFIG = (
        "C:\\faro\\pertzlab_mic_configs\\micromanager\\Moench\\TiMoench.cfg"
    )
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True
    DMD_NEEDS_TO_BE_WAKEN = True
    DMD_CHANNEL_GROUP = "TTL_ERK"
    # Manual config->(device, property) power mappings. These must be declared
    # explicitly: this config selects LED lines via numeric NIDAQ TTL states
    # (the preset stores e.g. State "16"; the color "GreenYellow" only lives in
    # the port0 state labels), so there's no reliable way to infer them. A
    # PowerChannel whose config is missing here raises in resolve_power()
    # instead of silently dropping the requested power.
    #
    # Derived from TiMoench.cfg: each preset sets NIDAQDO-Dev1/port0 State, and
    # the port's state labels give the color -> Spectra <Color>_Level:
    #   State  4 Cyan        -> Cyan_Level
    #   State 16 GreenYellow -> Green_Level
    #   State 32 Red         -> Red_Level
    #   State  8 Teal        -> Teal_Level
    POWER_PROPERTIES = {
        "CyanStim": ("LED", "Cyan_Level"),   # state 4  (pre-existing, known good)
        "mScarlet3": ("LED", "Green_Level"),  # state 16, confirmed on scope 2026-06-22
        "miRFP": ("LED", "Red_Level"),        # state 32, inferred from cfg labels
        "mCitrine": ("LED", "Teal_Level"),    # state 8,  inferred from cfg labels
    }
    # DMD calibration light path. The (device, property) for the power come
    # from POWER_PROPERTIES["CyanStim"] via resolve_power, so they aren't
    # duplicated here.
    DMD_CALIBRATION_CHANNEL = PowerChannel(
        config="CyanStim", group="TTL_ERK", power=10
    )
    BINNING = "2x2"
    # ROI applied as-is after binning — no centering recomputation.
    # Defaults below are for Prime BSI under 2x2 binning (1024 binned px wide).
    ROI_X = 0
    ROI_Y = 30
    ROI_WIDTH = 1024
    ROI_HEIGHT = 792
    SET_ROI_REQUIRED = True

    # Devices whose Busy() flag is unreliable — waitForDevice on these
    # eats the full 5 s MMCore timeout on every MDA event. Mosaic3 (DMD)
    # has the same stuck-Busy pathology as TIXYDrive; displaySLMImage()
    # commits the pattern synchronously before we reach the wait, so
    # skipping the poll is safe. See TODO.md #1.
    SKIP_WAIT_DEVICES: tuple[str, ...] = ("Mosaic3",)

    def __init__(self, affine_calibration_matrix=None):
        super().__init__()

        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus(mm_path=self.MICROMANAGER_PATH)
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

        self.affine_calibration_matrix = affine_calibration_matrix
        self.wakeup_dmd = None
        self.dmd_needs_to_be_waken = self.DMD_NEEDS_TO_BE_WAKEN
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        # Pin camera binning before set_roi(): MM camera drivers reset
        # the ROI on a binning change, so binning must come first.
        self.mmc.setConfig("Binning", self.BINNING)
        if self.SET_ROI_REQUIRED:
            self.set_roi()
        self.register_engine()

        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.dmd = DMD(
            self.mmc,
            self.DMD_CALIBRATION_CHANNEL,
            resolve_power=self.resolve_power,
            affine_matrix=self.affine_calibration_matrix,
        )
        self.wakeup_dmd = KeepDMDAlive(self.mmc, self.dmd)
        self.wakeup_dmd.run()

        self.image_height = self.mmc.getImageHeight()
        self.image_width = self.mmc.getImageWidth()

    def calibrate_dmd(
        self,
        verbose=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
        background=True,
    ):
        """Calibrate the DMD against the camera (if not already calibrated).

        Args:
            background: When True (default), the calibration MDAs run on a
                worker thread while this call pumps the Qt event loop, so
                napari stays responsive and previews the calibration spots
                live. The call still blocks until calibration finishes --
                it just doesn't freeze the GUI. Set False to run
                synchronously on the calling thread.

        Note:
            With ``background=True`` and ``verbose=True`` the matplotlib
            diagnostic plots are created from the worker thread. With the
            Jupyter inline backend this routes to the running cell fine;
            if plots misbehave, use ``background=False`` for verbose runs.
        """
        self.disable_log_output()

        if self.dmd is None or self.dmd.affine is not None:
            return

        def _do_calibration() -> None:
            self.wakeup_dmd.stop()
            try:
                self.dmd.calibrate(
                    verbose=verbose,
                    n_points=n_points,
                    radius=radius,
                    exposure=exposure,
                    marker_style=marker_style,
                    calibration_points_DMD=calibration_points_DMD,
                )
            finally:
                self.wakeup_dmd.run()

        if not background:
            _do_calibration()
            return

        # Run on a worker thread; pump Qt here so napari keeps repainting
        # and previewing the calibration frames. The call still blocks
        # until calibration is done -- it just doesn't starve the GUI.
        done = threading.Event()
        box: list[BaseException] = []

        def _worker() -> None:
            try:
                _do_calibration()
            except BaseException as exc:  # noqa: BLE001
                box.append(exc)
            finally:
                done.set()

        threading.Thread(
            target=_worker, name="DMDCalibration", daemon=True
        ).start()
        while not done.wait(timeout=0.05):
            _pump_qt_events()
        if box:
            raise box[0]

    def set_roi(self):
        """Apply the class's ROI_* settings as-is (after binning is set)."""
        self.mmc.clearROI()
        self.mmc.setROI(self.ROI_X, self.ROI_Y, self.ROI_WIDTH, self.ROI_HEIGHT)

    def post_experiment(self):
        """Post-process the experiment."""
        pass

    def shutdown(self):
        """Tear down hardware state so the microscope can be discarded.

        Stops the DMD wakeup loop and unloads all Micro-Manager devices
        so COM ports (notably the LED on COM3) and the SLM handle are
        released. Without this, pymmcore's native threads keep the
        Python process alive after the main thread exits, leaving a
        zombie that blocks the next session with
        ``Error in device "COM3"`` when MM tries to initialize.

        Idempotent with the atexit hook registered in
        :class:`PyMMCoreMicroscope`: calling ``shutdown`` explicitly just
        runs the same teardown earlier; if it's never called, the hook
        runs it at interpreter exit.
        """
        self._teardown_hardware()

    def _teardown_hardware(self) -> None:
        """Stop the DMD wakeup thread, then delegate to the base teardown.

        The wakeup thread keeps a reference to the SLM device; stopping
        it before ``unloadAllDevices`` avoids the unload racing the
        thread's next ``displaySLMImage`` call.
        """
        wakeup = getattr(self, "wakeup_dmd", None)
        if wakeup is not None:
            try:
                wakeup.stop()
            except Exception:
                pass
        super()._teardown_hardware()

    def register_engine(self, force: bool = False) -> None:
        """Create and register the microscope-specific MDA engine.

        This is idempotent unless `force=True`. It will attach a weakref to
        this microscope on the engine and register the engine on `self.mmc.mda`.
        """
        # If engine already exists and caller doesn't want to force, do nothing
        if hasattr(self, "engine") and self.engine is not None and not force:
            return

        # Create the engine and attach this microscope (weakref)
        self.engine = MoenchMDAEngine(self.mmc)
        try:
            self.engine.attach_microscope(self)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to attach microscope to engine"
            )

        # Register it on the MDARunner so acquisitions use it
        try:
            self.mmc.mda.set_engine(self.engine)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to register MDA engine on mmc.mda"
            )

    def disable_log_output(self):
        pymmcore_plus.configure_logging(
            stderr_level="CRITICAL",
            file_level="CRITICAL",
        )
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                for h in logger.handlers[:]:
                    logger.removeHandler(h)

        pymmcore_plus.configure_logging(stderr_level="WARNING")


class MoenchMDAEngine(MDAEngine):
    """Microscope-specific MDA engine for Moench.

    Override `setup_single_event` to add pre/post hooks for per-microscope
    behavior while preserving the base MDAEngine functionality by calling
    `super().setup_single_event(event)`.
    """

    def __init__(
        self,
        mmc,
        *,
        use_hardware_sequencing: bool = True,
        restore_initial_state: Optional[bool] = None,
    ):
        super().__init__(
            mmc,
            use_hardware_sequencing=use_hardware_sequencing,
            restore_initial_state=restore_initial_state,
        )
        self._microscope_ref: Optional[weakref.ref] = None
        self._log = logging.getLogger(self.__class__.__name__)

    def attach_microscope(self, mic) -> None:
        """Attach the microscope instance (weakref) so engine can consult it."""
        self._microscope_ref = weakref.ref(mic)

    @property
    def microscope(self):
        return None if self._microscope_ref is None else self._microscope_ref()

    def _set_event_channel(self, event: MDAEvent, max_retry_attempts: int = 5) -> None:
        if (ch := event.channel) is None:
            return

        # comparison with _last_config is a fast/rough check ... which may miss subtle
        # differences if device properties have been individually set in the meantime.
        # could also compare to the system state, with:
        # data = self._mmc.getConfigData(ch.group, ch.config)
        # if self._mmc.getSystemStateCache().isConfigurationIncluded(data):
        #     ...
        if (ch.group, ch.config) != self.mmcore._last_config:  # noqa: SLF001
            # Try multiple times to set the configuration in case of transient failures.
            for attempt in range(1, max_retry_attempts + 1):
                try:
                    self.mmcore.setConfig(ch.group, ch.config)
                except Exception as e:
                    logger.warning(
                        "Failed to set channel (attempt %d/%d). %s",
                        attempt,
                        max_retry_attempts,
                        e,
                    )
                    print(
                        "Failed to set channel (attempt %d/%d). %s",
                        attempt,
                        max_retry_attempts,
                        e,
                    )
                    if attempt == max_retry_attempts:
                        logger.warning(
                            "Giving up after %d attempts to set channel.",
                            max_retry_attempts,
                        )
                    else:
                        time.sleep(0.1)
                else:
                    break

    def _set_event_xy_position(self, event: MDAEvent, max_retry_attempts=5) -> None:
        event_x, event_y = event.x_pos, event.y_pos
        # If neither coordinate is provided, do nothing.
        if event_x is None and event_y is None:
            return

        core = self.mmcore
        # skip if no XY stage device is found
        if not core.getXYStageDevice():
            logger.warning("No XY stage device found. Cannot set XY position.")
            return

        # Retrieve the last commanded XY position.
        last_x, last_y = core._last_xy_position.get(None) or (
            None,
            None,
        )  # noqa: SLF001
        if (
            not self.force_set_xy_position
            and (event_x is None or event_x == last_x)
            and (event_y is None or event_y == last_y)
        ):
            return

        if event_x is None or event_y is None:
            cur_x, cur_y = core.getXYPosition()
            event_x = cur_x if event_x is None else event_x
            event_y = cur_y if event_y is None else event_y

        for attempt in range(0, max_retry_attempts):
            print
            try:
                core.setXYPosition(event_x, event_y)
                return
            except Exception as e:
                msg = str(e)
                if 'Wait for device "TIXYDrive" timed out' in msg:
                    if attempt == max_retry_attempts:
                        # all retries used, re-raise
                        raise
                    print(
                        f"[WARN] TIXYDrive wait timed out (attempt {attempt+1}/{max_retry_attempts+1}); "
                        f"retrying in {1} s..."
                    )
                    time.sleep(1)
                else:
                    # different error -> don't hide it
                    logger.warning("Failed to set XY position. %s", e)
                    raise

    def _wait_for_system_excluding_xy(self, event: MDAEvent) -> None:
        """Wait for all devices except TIXYDrive, then handle XY separately.

        TIXYDrive's Busy() flag is perpetually stuck on this microscope,
        so including it in waitForSystem() wastes 5s per event. Instead we
        wait for each device individually and only check XY position when
        a move was actually commanded. Devices listed in the microscope's
        SKIP_WAIT_DEVICES are bypassed for the same reason.
        """
        core = self.mmcore
        xy_stage = core.getXYStageDevice() if core.getXYStageDevice() else None

        skip = {"Core"}
        if xy_stage:
            skip.add(xy_stage)
        mic = self.microscope
        if mic is not None:
            skip.update(getattr(mic, "SKIP_WAIT_DEVICES", ()))

        # Wait for every loaded device except the XY stage and any
        # caller-declared skip devices.
        for dev in core.getLoadedDevices():
            if dev in skip:
                continue
            try:
                core.waitForDevice(dev)
            except RuntimeError as e:
                if "timed out" in str(e):
                    print(f"[WARN] waitForDevice({dev}) timed out ({e}), continuing.")
                else:
                    raise

        # Handle TIXYDrive: only wait if an XY move was commanded.
        # Since Busy() is perpetually stuck, we poll position directly
        # instead of relying on waitForDevice().
        target_xy = (event.x_pos, event.y_pos)
        if xy_stage and target_xy != (None, None):
            xy_tolerance_um = 1.0
            max_wait_s = 5.0
            poll_interval_s = 0.5
            elapsed = 0.0
            while elapsed < max_wait_s:
                try:
                    actual_xy = core.getXYPosition()
                    dx = abs(actual_xy[0] - target_xy[0])
                    dy = abs(actual_xy[1] - target_xy[1])
                    if dx < xy_tolerance_um and dy < xy_tolerance_um:
                        break
                except Exception:
                    pass
                time.sleep(poll_interval_s)
                elapsed += poll_interval_s
            else:
                try:
                    actual_xy = core.getXYPosition()
                except Exception:
                    actual_xy = "unknown"
                print(
                    f"[WARN] {xy_stage} not at target after {max_wait_s}s. "
                    f"target={target_xy}, actual={actual_xy}"
                )

    def setup_sequence(self, sequence):
        """Pause KeepDMDAlive for the duration of the MDA.

        The engine drives the DMD on every event (stim mask on stim
        frames, all-on on imaging frames when
        ``dmd_needs_to_be_waken``), so the 60 s background refresh is
        redundant during a run and just adds SLM-device contention.
        Restarted in ``teardown_sequence``.
        """
        mic = self.microscope
        if mic is not None:
            wakeup = getattr(mic, "wakeup_dmd", None)
            if wakeup is not None:
                try:
                    wakeup.stop()
                except Exception:
                    self._log.exception("Failed to stop wakeup_dmd before MDA")
        return super().setup_sequence(sequence)

    def teardown_sequence(self, sequence) -> None:
        super().teardown_sequence(sequence)
        mic = self.microscope
        if mic is not None:
            wakeup = getattr(mic, "wakeup_dmd", None)
            if wakeup is not None:
                try:
                    wakeup.run()
                except Exception:
                    self._log.exception("Failed to restart wakeup_dmd after MDA")

    def setup_event(self, event: MDAEvent) -> None:
        """Override to wait for devices individually, bypassing TIXYDrive.

        The TIXYDrive on this microscope has a perpetually-stuck Busy() flag,
        so waitForSystem() always times out after 5s. Instead, we wait for each
        device individually and handle TIXYDrive separately only when an XY
        move was actually commanded.
        """
        event = self._maybe_inject_dmd_wake_slm(event)
        if isinstance(event, SequencedEvent):
            self.setup_sequenced_event(event)
        else:
            self.setup_single_event(event)

        self._wait_for_system_excluding_xy(event)

    def exec_event(self, event: MDAEvent):
        """Override to inject the all-on SLM on non-stim events.

        Must mirror ``setup_event``'s injection: ``MDARunner`` keeps its
        own reference to the original event, so a ``model_copy`` inside
        ``setup_event`` doesn't propagate. Without this override
        ``_exec_event_slm_image`` (the ``displaySLMImage`` that arms the
        pattern for the next camera TTL) never fires for non-stim
        events, leaving the previously latched stim pattern to pulse
        instead.
        """
        event = self._maybe_inject_dmd_wake_slm(event)
        yield from super().exec_event(event)

    def _maybe_inject_dmd_wake_slm(self, event: MDAEvent) -> MDAEvent:
        """Hold the DMD all-on for non-stim captures.

        Under OverlapMode=On the DMD re-pulses its currently loaded
        pattern on every camera TTL. After a stim event the stim
        pattern stays latched, so subsequent imaging events at the
        same timepoint (e.g. other FOVs in an FOV-batched burst)
        would pulse the stim pattern instead of an all-on frame and
        come back dark. KeepDMDAlive's 60 s refresh is too slow to
        catch that burst.
        """
        if event.slm_image is not None:
            return event
        mic = self.microscope
        if mic is None or not getattr(mic, "dmd_needs_to_be_waken", False):
            return event
        dmd = getattr(mic, "dmd", None)
        if dmd is None:
            return event
        if event.metadata.get("img_type") == ImgType.IMG_STIM:
            return event
        return event.model_copy(
            update={
                "slm_image": SLMImage(
                    data=True, device=dmd.name, exposure=event.exposure
                )
            }
        )

    def setup_single_event(self, event: MDAEvent) -> None:
        """Setup hardware for a single (non-sequenced) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        if event.keep_shutter_open:
            ...

        max_retry_attempts = 10

        self._set_event_xy_position(event, max_retry_attempts=max_retry_attempts)

        if event.x_pos is not None or event.y_pos is not None:
            time.sleep(
                0.2
            )  # small delay to ensure XY stage has moved, as XY stage encore is broken on this microscope
        if event.z_pos is not None:
            self._set_event_z(event)
        if event.slm_image is not None:
            self._set_event_slm_image(event)

        self._set_event_channel(event, max_retry_attempts=max_retry_attempts)

        mmcore = self.mmcore
        if event.exposure is not None:
            try:
                mmcore.setExposure(event.exposure)
            except Exception as e:
                logger.warning("Failed to set exposure. %s", e)
        if event.properties is not None:
            for attempt in range(1, max_retry_attempts + 1):
                try:
                    for dev, prop, value in event.properties:
                        mmcore.setProperty(dev, prop, value)
                except Exception as e:
                    logger.warning("Failed to set properties. %s", e)
                    print(("Failed to set properties. %s", e))
                    if attempt == max_retry_attempts:
                        logger.warning(
                            "Giving up after %d attempts to set channel.",
                            max_retry_attempts,
                        )
                    else:
                        time.sleep(0.1)
                else:
                    break
        if (
            # (if autoshutter wasn't set at the beginning of the sequence
            # then it never matters...)
            self._autoshutter_was_set
            # if we want to leave the shutter open after this event, and autoshutter
            # is currently enabled...
            and event.keep_shutter_open
            and mmcore.getAutoShutter()
        ):
            # we have to disable autoshutter and open the shutter
            mmcore.setAutoShutter(False)
            mmcore.setShutterOpen(True)

    def _load_sequenced_event(
        self, event: SequencedEvent, max_retry_attempts: int = 0
    ) -> None:
        """Load a `SequencedEvent` into the core.

        `SequencedEvent` is a special pymmcore-plus specific subclass of
        `useq.MDAEvent`.
        """
        core = self.mmcore
        if event.exposure_sequence:
            cam_device = core.getCameraDevice()
            with suppress(RuntimeError):
                core.stopExposureSequence(cam_device)
            core.loadExposureSequence(cam_device, event.exposure_sequence)
        if event.x_sequence:  # y_sequence is implied and will be the same length
            stage = core.getXYStageDevice()
            with suppress(RuntimeError):
                core.stopXYStageSequence(stage)
            core.loadXYStageSequence(stage, event.x_sequence, event.y_sequence)
        if event.z_sequence:
            zstage = core.getFocusDevice()
            with suppress(RuntimeError):
                core.stopStageSequence(zstage)
            core.loadStageSequence(zstage, event.z_sequence)
        if event.slm_sequence:
            slm = core.getSLMDevice()
            with suppress(RuntimeError):
                core.stopSLMSequence(slm)
            core.loadSLMSequence(slm, event.slm_sequence)  # type: ignore[arg-type]
        if event.property_sequences:
            for (dev, prop), value_sequence in event.property_sequences.items():
                with suppress(RuntimeError):
                    core.stopPropertySequence(dev, prop)
                core.loadPropertySequence(dev, prop, value_sequence)

        # set all static properties, these won't change over the course of the sequence.
        if event.properties:
            for dev, prop, value in event.properties:
                for attempt in range(1, max_retry_attempts + 1):
                    try:
                        core.setProperty(dev, prop, value)
                    except Exception as e:
                        logger.warning(
                            "Failed to set property %s.%s (attempt %d/%d): %s",
                            dev,
                            prop,
                            attempt,
                            max_retry_attempts,
                            e,
                        )
                        if attempt == max_retry_attempts:
                            logger.warning(
                                "Giving up after %d attempts to set property %s.%s",
                                max_retry_attempts,
                                dev,
                                prop,
                            )
                        else:
                            time.sleep(0.1)
                    else:
                        break

    def setup_sequenced_event(
        self, event: SequencedEvent, max_retry_attempts: int = 5
    ) -> None:
        """Setup hardware for a sequenced (triggered) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        core = self.mmcore

        self._load_sequenced_event(event, max_retry_attempts=max_retry_attempts)

        # this is probably not necessary.  loadSequenceEvent will have already
        # set all the config properties individually/manually.  However, without
        # the call below, we won't be able to query `core.getCurrentConfig()`
        # not sure that's necessary; and this is here for tests to pass for now,
        # but this could be removed.
        self._set_event_channel(event, max_retry_attempts=max_retry_attempts)

        if event.slm_image:
            self._set_event_slm_image(event)

        # preparing a Sequence while another is running is dangerous.
        if core.isSequenceRunning():
            self._await_sequence_acquisition()
        core.prepareSequenceAcquisition(core.getCameraDevice())

        # start sequences or set non-sequenced values
        if event.x_sequence:
            core.startXYStageSequence(core.getXYStageDevice())
        else:
            self._set_event_xy_position(event)

        if event.z_sequence:
            core.startStageSequence(core.getFocusDevice())
        elif event.z_pos is not None:
            self._set_event_z(event)

        if event.exposure_sequence:
            core.startExposureSequence(core.getCameraDevice())
        elif event.exposure is not None:
            core.setExposure(event.exposure)

        if event.property_sequences:
            for dev, prop in event.property_sequences:
                core.startPropertySequence(dev, prop)
