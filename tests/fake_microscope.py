"""Drop-in ``PyMMCoreMicroscope`` driven by :class:`FakeCMMCorePlus`.

Usage::

    scene = MyScene(...)
    mic = FakeMicroscope(scene)
    ctrl = Controller(mic, pipeline)
    ctrl.run_experiment(events)

Scene-specific state lives on ``mic.scene``. Routes the Controller
through the normal pymmcore-plus path so ``mmc.run_mda``, the
``frameReady`` signal chain, and SLM dispatch all get test coverage.
"""

from __future__ import annotations

from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.microscope.simulation import SimDMD

from tests.fake_mmc import FakeCMMCorePlus, Scene


class FakeMicroscope(PyMMCoreMicroscope):
    """``PyMMCoreMicroscope`` backed by ``FakeCMMCorePlus`` + a scene.

    If the scene declares an SLM (``slm_name`` is truthy), a
    :class:`SimDMD` is attached so the stim branch of the Controller
    runs and the scene's ``on_slm_displayed`` fires with the dispatched
    mask.
    """

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        self.scene = scene
        self.mmc = FakeCMMCorePlus(scene)
        if getattr(scene, "slm_name", None):
            self.dmd = SimDMD(scene.slm_name)

    def init_scope(self) -> None:
        pass

    def post_experiment(self) -> None:
        pass
