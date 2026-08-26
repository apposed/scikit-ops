"""StarDist 2D inference, pretrained or from a model on disk.

Ported from src/imgops/implementations/stardist2d.py. Shares the
'stardist-tf' environment with starfun3d.py.

Two ops for the pretrained models rather than one with a ``model`` switch,
because they do not accept the same thing: the H&E model is trained on stain
colour and always wants a channel axis, while the fluorescence model wants a
single channel and collapses one if it is given it. That is a difference in
what the op *consumes*, so it belongs in ``Axes`` where a front end can see
it, not in an enum parameter where it is invisible until StarDist raises.

``stardist2d_custom`` is the third, and takes a directory. It cannot make the
same declaration -- a model on disk is single- or multi-channel depending on
how it was trained -- so it declares ``c?`` and reads the model's own
``config.json`` to decide. A caller with many trained models points this one
op at each of them in turn; see design 0011.
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


@op(env="stardist-tf")
def stardist2d_custom(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    model_dir: Annotated[str, {"widget_type": "FileEdit", "mode": "d"}] = "",
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
    """Detect objects with a StarDist 2D model loaded from disk.

    The counterpart to the two pretrained ops: same prediction, but the model
    comes from a directory rather than StarDist's download cache. That is what
    a model trained by ``skop.ops.train`` -- or by anything else that writes a
    StarDist model -- is loaded with.

    Args:
        image: Plane to segment. The trailing channel axis is collapsed when
            the model expects a single channel, and kept when it does not;
            the model's own config decides, not the caller.
        model_dir: Directory holding ``config.json`` and the weights. The
            directory name is the model name, as StarDist stores it.
        prob_thresh: Object probability threshold.
        nms_thresh: Non-maximum suppression threshold.
        normalize: Whether to percentile-normalize the input first.

    Returns:
        A label image, one integer per detected object.
    """
    import json
    import os

    from stardist.models import StarDist2D

    if not model_dir:
        raise ValueError(
            "model_dir is required -- this op loads a model from disk. Use "
            "stardist2d_fluo or stardist2d_he for the pretrained models."
        )

    model_dir = os.path.normpath(model_dir)
    basedir, name = os.path.split(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"No StarDist model in {model_dir}: no config.json")

    with open(config_path) as f:
        n_channel_in = json.load(f).get("n_channel_in", 1)

    x = image if n_channel_in > 1 else to_gray(image)

    progress(f"Loading model {name}")
    net = StarDist2D(None, name=name, basedir=basedir)

    return _run(net, x, prob_thresh, nms_thresh, normalize)


def _predict(
    x: np.ndarray,
    model: str,
    prob_thresh: float,
    nms_thresh: float,
    normalize: bool,
) -> np.ndarray:
    """Run one pretrained StarDist 2D model over an already-shaped array."""
    from stardist.models import StarDist2D

    progress(f"Loading pretrained model {model}")
    net = StarDist2D.from_pretrained(model)
    return _run(net, x, prob_thresh, nms_thresh, normalize)


def _run(
    net,
    x: np.ndarray,
    prob_thresh: float,
    nms_thresh: float,
    normalize: bool,
) -> np.ndarray:
    """Normalize if asked, then predict instances with an already-loaded net."""
    from csbdeep.utils import normalize as normalize_percentile

    if normalize:
        x = normalize_percentile(x, 1, 99.8)

    progress("Predicting instances")
    labels, _ = net.predict_instances(
        x,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    return np.asarray(labels).astype(np.uint16)
