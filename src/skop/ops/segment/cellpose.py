"""Cellpose segmentation.

Ported from src/imgops/implementations/cellpose.py.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, Extra, op, progress
from skop.types import ImageData, LabelsData

from .._util import to_gray


@op(env="cellpose")
def cellpose(
    image: Annotated[ImageData, Axes("yxc?", extra=Extra.iterate)],
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
    use_gpu: bool = True,
) -> LabelsData:
    """Segment cells with Cellpose.

    Args:
        image: Plane to segment. A trailing RGB(A) axis is collapsed. A
            caller naming its axes may hand this a stack instead.
        diameter: Expected cell diameter in pixels; 0 lets Cellpose estimate.
        flow_threshold: Maximum allowed flow error per mask.
        cellprob_threshold: Cell probability cutoff; lower finds more cells.
        use_gpu: Whether to use the GPU, when one is available.

    Returns:
        A label image, one integer per detected cell.
    """
    from cellpose import models

    gray = to_gray(image)

    progress("Loading Cellpose model")
    model = models.CellposeModel(gpu=use_gpu)

    progress("Running Cellpose")
    result = model.eval(
        gray,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # NB: eval returns (masks, flows, styles) or (masks, flows, styles, diams),
    # depending on the Cellpose version.
    return np.asarray(result[0]).astype(np.uint16)
