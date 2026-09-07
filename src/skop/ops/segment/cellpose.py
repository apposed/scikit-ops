"""Cellpose segmentation, on the CellposeSAM model of version 4.

Ported from src/imgops/implementations/cellpose.py.

Runs in the shared 'pytorch' environment rather than one of its own. Cellpose
4 is a maintained conda package that solves alongside ultralytics and
micro_sam, so it is exactly the occupant that environment exists for (0002):
one build, one warm worker, shared by several ops.

Cellpose 3 is the exception, in ``cellpose3.py`` and an environment of its
own -- version 4 replaced the model zoo and moved the API, so no pin makes
them coexist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData

from .._util import channel_axis, to_gray


@op(env="pytorch")
def cellpose(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    pretrained_model: Path | None = None,
    diameter: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 1.0},
    ] = 0.0,
    flow_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 3.0, "step": 0.05},
    ] = 0.4,
    cellprob_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": -6.0, "max": 6.0, "step": 0.1},
    ] = 0.0,
    niter: int = 0,
    min_size: int = 15,
    normalize: bool = True,
    collapse_channels: bool = False,
    use_gpu: bool = True,
) -> LabelsData:
    """Segment cells with Cellpose.

    Args:
        image: Plane to segment. A trailing RGB(A) axis is passed through
            to CPSAM, which reads up to three channels itself. A caller
            naming its axes may hand this a stack instead.
        pretrained_model: A CPSAM model finetuned on your own data. Empty
            runs the built-in CPSAM. This must be a CPSAM model, not one
            from Cellpose 3 -- those load in ``cellpose3`` instead, and
            ``skop.models.cellpose_flavor`` tells the two apart from the
            file.
        diameter: Expected cell diameter in pixels; 0 lets Cellpose estimate.
        flow_threshold: Maximum allowed flow error per mask.
        cellprob_threshold: Cell probability cutoff; lower finds more cells.
        niter: Dynamics iterations; 0 lets Cellpose scale it to the diameter.
            More is slower and helps long or branched cells, whose pixels
            need further to travel.
        min_size: Discard masks smaller than this many pixels. -1 keeps
            everything, which is what you want when the objects are small
            enough that the default is doing the discarding for you.
        normalize: Whether to percentile-normalize the plane first. Cellpose
            does this per call, so a caller running it slice by slice over a
            stack gets each plane stretched to its own range -- which turns a
            faint plane at the top of a volume into a bright one full of
            detections. Turn it off and normalize the volume beforehand when
            that matters; see ``skop.ops.workflows.segment``.
        collapse_channels: Average a trailing RGB(A) axis to grey before
            segmenting. Off, because CPSAM takes 1-3 channels in any order
            and using them beats throwing them away. On is for when one
            colour is noise the average dilutes -- and for matching
            ``cellpose3``, which needs the channels named or collapsed.
        use_gpu: Whether to use the GPU, when one is available.

    Returns:
        A label image, one integer per detected cell.
    """
    from cellpose import models

    if collapse_channels:
        plane, axis = to_gray(image), None
    else:
        plane, axis = np.asarray(image), channel_axis(image)

    # Passed only when set, rather than relying on what the version in this
    # environment treats as "no model" -- it has been None and False.
    weights = {}
    if pretrained_model is not None:
        weights["pretrained_model"] = str(pretrained_model)
        progress(f"Loading Cellpose model {Path(pretrained_model).name}")
    else:
        progress("Loading Cellpose model")
    model = models.CellposeModel(gpu=use_gpu, **weights)

    progress("Running Cellpose")
    result = model.eval(
        plane,
        channel_axis=axis,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        niter=niter if niter > 0 else None,
        min_size=min_size,
        normalize=normalize,
    )
    # NB: eval returns (masks, flows, styles) or (masks, flows, styles, diams),
    # depending on the Cellpose version.
    return np.asarray(result[0]).astype(np.uint16)
