"""
Experiment: StarDist segmenter as a magicgui @guiclass "command".

Minimal stand-in for the dataclass-based StardistSegmenter -- 2D only,
pretrained models only, no training.  Uses magicgui's ``@guiclass`` so
the parameter GUI is generated automatically.

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


class StardistModel(Enum):
    """Pretrained 2D StarDist models."""

    fluo = "2D_versatile_fluo"
    he = "2D_versatile_he"


@guiclass
class StardistCommand:
    """StarDist 2D inference with a pretrained model."""

    NAME = "StarDist (2D)"

    model: StardistModel = StardistModel.fluo
    prob_thresh: float = 0.5
    nms_thresh: float = 0.4
    normalize: bool = True

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run pretrained StarDist2D and return a uint16 label image."""
        from csbdeep.utils import normalize as _normalize
        from stardist.models import StarDist2D

        # 2D_versatile_he expects RGB; the fluo model expects grayscale.
        if self.model is StardistModel.fluo:
            x = _to_gray(image)
        else:
            x = image

        if self.normalize:
            x = _normalize(x, 1, 99.8)

        model = StarDist2D.from_pretrained(self.model.value)
        labels, _ = model.predict_instances(
            x,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
        )
        return np.asarray(labels).astype(np.uint16)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis to grayscale."""
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    return image
