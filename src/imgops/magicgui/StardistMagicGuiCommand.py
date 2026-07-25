"""
StarDist segmenter as a magicgui @guiclass "command".

if depenendencies available runs stardist implementation directory.

otherwise falls back to running stardist via appose and a default stardist environment

Exposes the same tiny interface as the other experiment commands:
* ``.gui``            -- auto-generated magicgui Container widget.
* ``.segment(image)`` -- run segmentation, return a uint16 label image.
* ``NAME``            -- label shown in the switcher combo.

stardist is imported lazily inside ``segment`` so this file (and the
switcher panel) still load in environments without stardist installed.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from magicgui.experimental import guiclass

# Try to import stardist at module level
from imgops.implementations.stardist2d import run_stardist, env_name, dependencies_available 

class StardistModel(Enum):
    """Pretrained 2D StarDist models."""

    fluo = "2D_versatile_fluo"
    he = "2D_versatile_he"

@guiclass
class StardistCommand:
    """StarDist 2D inference with a pretrained model."""

    NAME = "StarDist (2D)"

    # model: StardistModel = StardistModel.fluo
    prob_thresh: float = 0.5
    nms_thresh: float = 0.4
    normalize: bool = True

    def are_dependencies_available(self) -> bool:
        return dependencies_available

    def environment_name(self) -> str | None:
        """Return the pinned remote env path for this command, or None."""
        return env_name if env_name is not None else None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run pretrained StarDist2D and return a uint16 label image."""

        return  run_stardist(
            image,
            model_name="2D_versatile_fluo",
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
            normalize=self.normalize,
        )

    # -- appose remote execution ------------------------------------------

    def get_execution_inputs(self) -> dict:
        """Return member variables flattened to appose-safe primitives.

        The enum is reduced to its string ``.value`` and everything else
        is a plain float/bool. These are passed as task inputs so the
        worker script can rebuild what it needs without any reference to
        ``self`` or the ``StardistModel`` class.
        """
        return {
            "model_name": "2D_versatile_fluo",
            "prob_thresh": float(self.prob_thresh),
            "nms_thresh": float(self.nms_thresh),
            "normalize": bool(self.normalize),
        }

    def get_execution_string(self) -> str:
        """Return a self-contained worker script for appose.

        Reads the script from appose/workers/stardist2d_worker.py so the
        worker logic lives in one place and does not have to be kept in sync
        with this file.
        """
        from pathlib import Path
        worker_path = Path(__file__).resolve().parents[1] / "appose" / "workers" / "stardist2d_worker.py"
        return worker_path.read_text()

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis to grayscale."""
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    return image
