"""Regenerate the sample images embedded in ``tests/README.md``.

Run from the repo root::

    python -m tests.assets.generate

The images show what the built-in scenes produce so contributors have a
visual reference for what their tests are operating on.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from useq import MDAEvent

from faro.segmentation.base import SegmentatorBinary

from tests.fixtures import make_circle_image
from tests.test_tracking_accuracy import (
    StimPerCellCenter,
    SyntheticCellScene,
)


_HERE = os.path.dirname(os.path.abspath(__file__))


def _save(fig, name: str) -> None:
    path = os.path.join(_HERE, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def circle_scene() -> None:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(make_circle_image(), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("CircleScene", fontsize=10)
    fig.tight_layout()
    _save(fig, "scene_circle.png")


def synthetic_cells_strip() -> None:
    scene = SyntheticCellScene(n_cells=50, n_frames=15)
    gt = np.stack(scene.gt, axis=0)  # (T, N, 2) of (row, col)
    n_cells = gt.shape[1]
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), dpi=150)
    for ax, t in zip(axes, [0, 5, 10, 14]):
        ax.imshow(scene.render(MDAEvent(index={"t": t, "p": 0, "c": 0})), cmap="gray")
        if t > 0:
            # Draw the GT trail up to frame t for every cell, color per cell.
            for i in range(n_cells):
                rows = gt[: t + 1, i, 0]
                cols = gt[: t + 1, i, 1]
                ax.plot(cols, rows, color=cmap(i % 20), lw=0.8, alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t}", fontsize=11)
    fig.suptitle(
        "SyntheticCellScene (50 cells, drifting; GT tracks overlaid)", fontsize=12
    )
    fig.tight_layout()
    _save(fig, "scene_synthetic_cells.png")


def stim_mask_overlay() -> None:
    scene = SyntheticCellScene(n_cells=50, n_frames=15)
    cells = scene.render(MDAEvent(index={"t": 8, "p": 0, "c": 0}))
    labels = SegmentatorBinary().segment(cells)
    stim_mask, _ = StimPerCellCenter(radius=2).get_stim_mask({"labels": labels})

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=150)
    axes[0].imshow(cells, cmap="gray")
    axes[0].set_title("raw (t=8)")
    axes[1].imshow(labels, cmap="nipy_spectral")
    axes[1].set_title("segmentation")
    axes[2].imshow(cells, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(stim_mask == 0, stim_mask),
        cmap="Reds",
        alpha=0.9,
        vmin=0,
        vmax=255,
    )
    axes[2].set_title("stim mask overlay")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("StimPerCellCenter on SyntheticCellScene", fontsize=12)
    fig.tight_layout()
    _save(fig, "stim_mask_overlay.png")


if __name__ == "__main__":
    circle_scene()
    synthetic_cells_strip()
    stim_mask_overlay()
