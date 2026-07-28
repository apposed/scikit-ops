"""Cellpose 3, the model zoo that came before CellposeSAM.

Cellpose 4 replaced a shelf of specialised models -- cyto3, nuclei, and the
ones trained on bacteria, yeast and tissue -- with one CellposeSAM model that
is better on most things and not on everything. ``skop.ops.segment.cellpose``
is that one; this is the shelf, kept because a model trained on your organism
still beats a general one that never saw it, and because published results
were produced with these.

A separate op in a separate environment rather than a version parameter on
one op. The two are not API-compatible: v3's ``models.Cellpose`` became v4's
``models.CellposeModel``, and ``eval`` lost a return value on the way. The
old ``CellposeSegmenter`` in napari-ai-lab carried both, branching on
``cellpose.version`` at every call site -- which is exactly the runtime
version sniffing an environment per op is meant to replace. Pinning the
environment to ``cellpose <4`` means the code below has one path.

``models.Cellpose``, not ``models.CellposeModel``, is the important detail.
In v3 they are different classes: ``Cellpose`` bundles a SizeModel, so
``diameter=None`` makes it estimate the diameter before segmenting.
``CellposeModel`` has no such thing and would silently use its default. That
is also why v3's ``eval`` returns four values where v4 returns three -- the
fourth is the diameter it estimated.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData

from .._util import to_gray


class PretrainedModel(Enum):
    """Cellpose 3's built-in models, by what they were trained on.

    The generalist pair first. The rest are the specialists, each finetuned
    on one imaging domain and worth reaching for when the generalist misses
    -- which is the reason this op exists at all.
    """

    cyto3 = "cyto3"
    nuclei = "nuclei"
    bacteria_phase = "bact_phase_cp3"
    bacteria_fluorescence = "bact_fluor_cp3"
    yeast_phase = "yeast_PhC_cp3"
    yeast_brightfield = "yeast_BF_cp3"
    tissue = "tissuenet_cp3"
    live_cell = "livecell_cp3"
    deepbacs = "deepbacs_cp3"


@op(env="cellpose3")
def cellpose3(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    model: PretrainedModel = PretrainedModel.cyto3,
    diameter: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 1.0},
    ] = 30.0,
    flow_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 3.0, "step": 0.05},
    ] = 0.4,
    cellprob_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": -6.0, "max": 6.0, "step": 0.1},
    ] = 0.0,
    niter: int = 200,
    use_gpu: bool = True,
) -> LabelsData:
    """Segment cells with one of Cellpose 3's pretrained models.

    Args:
        image: Plane to segment. A trailing RGB(A) axis is collapsed. A
            caller naming its axes may hand this a stack instead, and it is
            segmented plane by plane.
        model: Which pretrained model to run. cyto3 is the generalist; the
            specialists are worth trying when it misses on your organism.
        diameter: Expected cell diameter in pixels. 0 lets Cellpose estimate
            it per image with its size model, which costs a second pass and
            is the right choice when object size varies between images.
        flow_threshold: Maximum allowed flow error per mask. Higher accepts
            more irregular shapes; lower discards them.
        cellprob_threshold: Cell probability cutoff. Lower finds more, and
            grows the masks it finds.
        niter: Dynamics iterations. More is slower and helps long or
            branched cells, whose pixels need further to travel.
        use_gpu: Whether to use the GPU when one is available. Falls back to
            CPU rather than failing if the GPU model cannot be built.

    Returns:
        A label image, one integer per detected cell.

    Note: weights are downloaded into Cellpose's own cache on first use.
    """
    from cellpose import models

    gray = to_gray(image)
    # int32 reaches eval from label-image arithmetic and upstream ops, and
    # Cellpose's normalization does not handle it. Carried over from
    # napari-ai-lab's CellposeSegmenter, which hit it.
    if gray.dtype == np.int32:
        gray = gray.astype(np.float32)

    progress(f"Loading Cellpose 3 model {model.value}")
    try:
        net = models.Cellpose(gpu=use_gpu, model_type=model.value)
    except (AttributeError, ValueError, TypeError) as exc:
        # A GPU that torch can see but Cellpose cannot use fails here rather
        # than at eval. Segmenting slowly beats not segmenting.
        progress(f"GPU model unavailable ({exc}); falling back to CPU")
        net = models.Cellpose(gpu=False, model_type=model.value)

    progress("Running Cellpose 3")
    # Four returns in v3 -- masks, flows, styles, diams -- where v4 gives
    # three. Indexed rather than unpacked so the difference cannot bite.
    result = net.eval(
        gray,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        niter=niter,
    )
    return np.asarray(result[0]).astype(np.uint16)
