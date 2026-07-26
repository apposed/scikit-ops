"""StarDist 2D inference with a pretrained model.

Ported from src/imgops/implementations/stardist2d.py. Shares the
'stardist-tf' environment with starfun3d.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import numpy as np

from skop import Axes, Extra, op, progress
from skop.types import ImageData, LabelsData

from .._util import to_gray


class PretrainedModel(Enum):
    """StarDist's pretrained 2D models."""

    fluo = "2D_versatile_fluo"
    he = "2D_versatile_he"


@op(env="stardist-tf")
def stardist2d(
    image: Annotated[ImageData, Axes("yxc?", extra=Extra.iterate)],
    model: PretrainedModel = PretrainedModel.fluo,
    prob_thresh: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.5,
    nms_thresh: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.4,
    normalize: bool = True,
) -> LabelsData:
    """Detect objects in a 2D image with a pretrained StarDist model.

    Args:
        image: Plane to segment. A caller naming its axes may hand this a
            stack instead, to be segmented one plane at a time.
        model: Which pretrained model to use. The fluorescence model wants a
            single channel; the H&E model wants RGB.
        prob_thresh: Object probability threshold.
        nms_thresh: Non-maximum suppression threshold.
        normalize: Whether to percentile-normalize the input first.

    Returns:
        A label image, one integer per detected object.

    Note: the model weights are downloaded on first use.
    """
    from csbdeep.utils import normalize as normalize_percentile
    from stardist.models import StarDist2D

    # The H&E model wants the colour channels the fluorescence model discards.
    x = to_gray(image) if model is PretrainedModel.fluo else image

    if normalize:
        x = normalize_percentile(x, 1, 99.8)

    progress(f"Loading pretrained model {model.value}")
    net = StarDist2D.from_pretrained(model.value)

    progress("Predicting instances")
    labels, _ = net.predict_instances(
        x,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    return np.asarray(labels).astype(np.uint16)
