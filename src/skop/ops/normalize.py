"""Intensity normalization, separated from the models that assume it.

Every deep learning segmenter wants its input scaled the way its training
data was, and most of them do it silently on whatever array they are handed.
That is fine until the array is a slice of something bigger.

StarDist is the case this exists for. ``stardist2d_fluo`` declares
``Axes("y", "x", "c?")``, so a volume handed to it is looped over plane by
plane -- and csbdeep normalizes *each plane* against its own percentiles. A
top or bottom plane holding nothing but out-of-focus blur gets stretched until
that blur fills the range and reads as nuclei, so the segmenter finds objects
there that are not there.

Normalizing first, over the whole volume, fixes it: the empty planes stay
dim relative to the ones with signal in them, because they are measured
against the same range. Hence ``Axes(variadic=True)`` here, the same
declaration ``otsu`` carries -- percentiles do not care how many axes they are
given, so the volume arrives whole and is normalized in one call, and only
then does the 2-D segmenter run slice by slice.

Pure numpy, so it lives in the 'minimal' environment and composes with a model
in any other. An op's environment is where it runs, not where its caller runs:
this normalizing in 'minimal' and feeding StarDist in 'stardist-tf' is two
worker processes and one shared-memory handoff.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData


@op(env="minimal")
def percentile(
    image: Annotated[ImageData, Axes(variadic=True)],
    low: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 100.0, "step": 0.1},
    ] = 1.0,
    high: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 100.0, "step": 0.1},
    ] = 99.8,
    clip: bool = False,
) -> ImageData:
    """Rescale intensities so that two percentiles land on 0 and 1.

    The normalization csbdeep applies inside StarDist and CARE, as an op that
    can be called on its own -- which matters whenever the thing consuming it
    works on fewer axes than you have. See this module's docstring.

    Percentiles rather than min and max, because one hot pixel or one dead
    one would otherwise set the whole range.

    Args:
        image: Image to normalize. Any number of axes; all of them are
            measured together, so a volume gets one range rather than one per
            plane. A trailing RGB axis is included in that, which is right for
            a photograph and wrong for unrelated fluorescence channels --
            normalize those separately for now.
        low: Percentile mapped to 0. StarDist's own default is 1.
        high: Percentile mapped to 1. StarDist's own default is 99.8.
        clip: Whether to cut everything outside [0, 1]. Off by default,
            matching csbdeep: values beyond the percentiles are real signal,
            and a model trained this way saw them.

    Returns:
        The image as float32, with ``low`` at 0 and ``high`` at 1.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"need 0 <= low < high <= 100, got low={low}, high={high}")

    array = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(array, [low, high])

    # The epsilon is csbdeep's, and it is what keeps a blank image -- every
    # pixel identical, so lo == hi -- from dividing by zero and returning nan.
    out = (array - lo) / (hi - lo + 1e-20)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)
