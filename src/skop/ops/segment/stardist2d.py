"""StarDist 2D inference with a pretrained model.

Ported from src/imgops/implementations/stardist2d.py. Shares the
'stardist-tf' environment with starfun3d.py.

Two ops rather than one with a ``model`` switch, because the two pretrained
models do not accept the same thing: the H&E model is trained on stain colour
and always wants a channel axis, while the fluorescence model wants a single
channel and collapses one if it is given it. That is a difference in what the
op *consumes*, so it belongs in ``Axes`` where a front end can see it, not in
an enum parameter where it is invisible until StarDist raises.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData

from .._util import to_gray

_FLUO = "2D_versatile_fluo"
_HE = "2D_versatile_he"


@op(env="stardist-tf")
def stardist2d_fluo(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
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
    """Detect objects in a fluorescence image with pretrained StarDist.

    Args:
        image: Plane to segment. A trailing RGB(A) axis is collapsed, since
            this model wants a single channel.
        prob_thresh: Object probability threshold.
        nms_thresh: Non-maximum suppression threshold.
        normalize: Whether to percentile-normalize the input first.

    Returns:
        A label image, one integer per detected object.

    Note: the model weights are downloaded on first use.
    """
    return _predict(to_gray(image), _FLUO, prob_thresh, nms_thresh, normalize)


@op(env="stardist-tf")
def stardist2d_he(
    image: Annotated[ImageData, Axes("y", "x", "c")],
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
    """Detect objects in an H&E-stained image with pretrained StarDist.

    Args:
        image: Plane to segment, with its colour channels intact -- this model
            is trained on stain colour, so the channel axis is required.
        prob_thresh: Object probability threshold.
        nms_thresh: Non-maximum suppression threshold.
        normalize: Whether to percentile-normalize the input first.

    Returns:
        A label image, one integer per detected object.

    Note: the model weights are downloaded on first use.
    """
    return _predict(image, _HE, prob_thresh, nms_thresh, normalize)


def _predict(
    x: np.ndarray,
    model: str,
    prob_thresh: float,
    nms_thresh: float,
    normalize: bool,
) -> np.ndarray:
    """Run one pretrained StarDist 2D model over an already-shaped array."""
    from csbdeep.utils import normalize as normalize_percentile
    from stardist.models import StarDist2D

    if normalize:
        x = normalize_percentile(x, 1, 99.8)

    progress(f"Loading pretrained model {model}")
    net = StarDist2D.from_pretrained(model)

    progress("Predicting instances")
    labels, _ = net.predict_instances(
        x,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    return np.asarray(labels).astype(np.uint16)
