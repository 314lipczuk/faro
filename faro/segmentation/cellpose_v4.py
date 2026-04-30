import threading
import warnings

warnings.filterwarnings("ignore", message="Sparse invariant checks")

import numpy as np
from faro.segmentation.base import Segmentator
import skimage
from cellpose import models


class CellposeV4(Segmentator):

    def __init__(
        self,
        custom_model_path=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
        gpu: bool = True,
    ):

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size

        if custom_model_path is None:
            self.model = models.CellposeModel(gpu=gpu)
        else:
            self.model = models.CellposeModel(
                pretrained_model=custom_model_path, gpu=gpu
            )
        # Analyzer.executor runs up to ``max_workers`` pipeline tasks at
        # once — under FOV batching that is one task per FOV in a batch
        # firing nearly simultaneously. ``CellposeModel.eval`` is not
        # thread-safe (shared internal buffers + a single CUDA context
        # under PyTorch), and overlapping calls have been observed to
        # silently return empty/corrupt masks for an entire batch of
        # FOVs while the next batch works. Serialize ``.eval`` here so
        # at most one segmentation runs at a time. Tracking, feature
        # extraction, and storage stay parallel across FOVs.
        self._eval_lock = threading.Lock()

    def segment(self, image: np.ndarray) -> np.ndarray:

        with self._eval_lock:
            masks, flows, styles = self.model.eval(
                image,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
            )

        if self.min_size > 0:
            # remove cells below threshold
            masks = skimage.morphology.remove_small_objects(
                masks, min_size=self.min_size, connectivity=1
            )
        return masks
