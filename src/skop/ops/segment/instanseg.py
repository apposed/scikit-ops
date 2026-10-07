"""InstanSeg segmentation: nuclei, and cells around them.

InstanSeg (instanseg-torch) is plain torch, so it shares the 'pytorch'
environment (0002). Its fluorescence model takes any number of channels, in
any order, and labels nuclei and whole cells in one pass.

The scripts InstanSeg ships with also read files -- Bio-Formats through a
JVM, whole-slide readers -- and none of that is here. The op is handed an
array; reading the file is the caller's job.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, NamedTuple

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData


class PretrainedModel(Enum):
    """InstanSeg's published models."""

    fluorescence_nuclei_and_cells = "fluorescence_nuclei_and_cells"
    brightfield_nuclei = "brightfield_nuclei"
    single_channel_nuclei = "single_channel_nuclei"


class Target(Enum):
    """What to label, for a model that finds both."""

    nuclei_and_cells = "all_outputs"
    nuclei = "nuclei"
    cells = "cells"


class NucleiAndCells(NamedTuple):
    #: One label per nucleus.
    nuclei: LabelsData | None
    #: One label per cell, numbered to match its nucleus. None for a model
    #: that only finds nuclei, or when only nuclei were asked for.
    cells: LabelsData | None


#: Bigger than this, in pixels along either axis, and the image is run in
#: tiles rather than whole.
_WHOLE = 2048


@op(env="pytorch")
def instanseg(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    cells: Annotated[ImageData, Axes("y", "x", "c?")] | None = None,
    model: PretrainedModel = PretrainedModel.fluorescence_nuclei_and_cells,
    target: Target = Target.nuclei_and_cells,
    nuclei_channel: Annotated[int, {"min": 0, "max": 64}] = 0,
    cells_channel: Annotated[int, {"min": 0, "max": 64}] = 0,
    pixel_size: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0, "step": 0.05},
    ] = 0.0,
    use_gpu: bool = True,
) -> NucleiAndCells:
    """Label nuclei and cells with InstanSeg.

    Args:
        image: Plane to segment: the nuclei, or every channel when *cells*
            is not given. A trailing channel axis is passed to the model,
            which takes any number of fluorescence channels; the brightfield
            model wants RGB.
        cells: The cells -- membrane or cytoplasm -- when they are a layer
            of their own. Goes to the model after the nuclei.
        model: Which published model to run. Downloaded on first use.
        target: Nuclei, cells or both, for the model that finds both.
        nuclei_channel: The nuclei's channel in *image*, counting from 1;
            0 for all of it.
        cells_channel: The cells' channel, counting from 1: in *cells* when
            it is given, else in *image*. 0 for all of *cells*, or with no
            *cells* and no nuclei channel either, all of *image*. So: two
            layers, 0 and 0; one layer of channels, given twice or once,
            the two channel numbers.
        pixel_size: The image's pixel size in microns. InstanSeg was trained
            at about 0.5, and rescales the image to that; 0 runs it at the
            size it is, which is right only if the pixels are near 0.5 um.
        use_gpu: Whether to use the GPU, when one is available.

    Returns:
        nuclei: Labels, one per nucleus.
        cells: Labels, one per cell, or None when there are none to give.

    Note: the weights are downloaded into skop's asset cache on first use.
    """
    import os

    import torch

    from skop.assets import cache_dir

    # InstanSeg downloads into its own package folder unless told otherwise.
    os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = str(cache_dir() / "instanseg")
    from instanseg import InstanSeg

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    progress(f"Loading {model.value} on {device}")
    net = InstanSeg(model.value, device=device, verbosity=0)

    def channels_first(array: np.ndarray) -> np.ndarray:
        # InstanSeg wants channels first; a plane with none gets one.
        return array[np.newaxis] if array.ndim == 2 else np.moveaxis(array, -1, 0)

    def pick(planes: np.ndarray, channel: int, name: str) -> np.ndarray:
        if not channel:
            return planes
        if channel > len(planes):
            raise ValueError(
                f"{name} channel {channel} asked for, but it has {len(planes)}; "
                "channels count from 1"
            )
        return planes[channel - 1 : channel]

    if cells is None:
        # One input: the channel numbers pick within it, nuclei first.
        planes = channels_first(image)
        chosen = list(dict.fromkeys(c for c in (nuclei_channel, cells_channel) if c))
        if chosen:
            planes = np.concatenate([pick(planes, c, "image") for c in chosen])
    else:
        if cells.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"cells is {cells.shape[0]} x {cells.shape[1]}, but image is "
                f"{image.shape[0]} x {image.shape[1]}"
            )
        planes = np.concatenate(
            [
                pick(channels_first(image), nuclei_channel, "image"),
                pick(channels_first(cells), cells_channel, "cells"),
            ]
        )
    planes = np.ascontiguousarray(planes, dtype=np.float32)
    kwargs = {
        "pixel_size": pixel_size or None,
        "target": target.value,
        "return_image_tensor": False,
    }

    progress(f"Segmenting {image.shape[0]} x {image.shape[1]}")
    if max(image.shape[:2]) <= _WHOLE:
        labels = net.eval_small_image(planes, **kwargs)
    else:
        labels = net.eval_medium_image(planes, tile_size=512, **kwargs)
    labels = labels.squeeze(0).cpu().numpy().astype(np.int32)

    # One plane per output: nuclei then cells for the model that finds both
    # and was asked for both, else just the one asked for.
    if len(labels) == 2:
        return NucleiAndCells(nuclei=labels[0], cells=labels[1])
    if target is Target.cells:
        return NucleiAndCells(nuclei=None, cells=labels[0])
    return NucleiAndCells(nuclei=labels[0], cells=None)
