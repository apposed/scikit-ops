"""Cellpose 4 segmentation, on its transformer models.

Four models, two backbones: CPSAM and the DINOv3 one added in June 2026.
Same call and same finetuning for all four, so one op with a model choice.

Shares the 'pytorch' environment (0002). Cellpose 3 cannot: version 4
replaced the zoo and moved the API, so it has an environment of its own.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData

from .._util import channel_axis, to_gray


class PretrainedModel(Enum):
    """Cellpose 4's built-ins, newest of each backbone first. cpdino_vitb is
    ViT-B, about a third the size of the others and cheaper to finetune."""

    cpsam_v2 = "cpsam_v2"
    cpsam = "cpsam"
    cpdino = "cpdino"
    cpdino_vitb = "cpdino-vitb"


@op(env="pytorch")
def cellpose4(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    model: PretrainedModel = PretrainedModel.cpsam_v2,
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
        model: Which built-in to run. Ignored when ``pretrained_model``
            is set.
        pretrained_model: A Cellpose 4 model finetuned on your own data.
            Empty runs the built-in named by ``model``. It must be a
            Cellpose 4 model, not one from Cellpose 3 -- those load in
            ``cellpose3`` instead -- and it carries its own backbone, so
            ``model`` does not choose one for it.
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
    # Cellpose 4 reads `pretrained_model` as a name or a path.
    if pretrained_model is not None:
        source = str(pretrained_model)
        progress(f"Loading Cellpose model {Path(pretrained_model).name}")
    else:
        source = model.value
        progress(f"Loading Cellpose model {source}")
    net = models.CellposeModel(gpu=use_gpu, pretrained_model=source)

    progress("Running Cellpose")
    result = net.eval(
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
