"""micro_sam automatic instance segmentation: every object, no prompts.

The whole-image counterpart of ``skop.ops.prompt.microsam`` (clicks) and
``skop.ops.mask.microsam`` (boxes). With a model that has micro_sam's
instance decoder -- ``vit_b_lm``, ``vit_l_lm``, ``vit_b_em_organelles`` and
anything ``skop.ops.train.microsam`` trained -- that is AIS, and fast; plain
``vit_b`` has none and falls back to SAM's automatic mask generator, which
is slow.

Shares the 'pytorch' environment (0002), which carries micro_sam.

A trained checkpoint does not say which micro_sam model it was finetuned
from, and loading it needs that ``model_type``. So the trainer writes
``skop_microsam.json`` beside ``best.pt``, and this reads it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.runtime import progress
from opspec.types import ImageOf, LabelsOf

#: The file, beside a trained best.pt, saying what it was finetuned from.
META = "skop_microsam.json"

#: One loaded model: (model_type, checkpoint, mtime, tiled) -> pair.
_loaded: dict = {}


def _ndim(shape) -> int:
    """2 for YX or YXC, 3 for ZYX -- ai-lab's get_ndim rule."""
    if len(shape) == 3 and shape[-1] < 5:
        return 2
    return len(shape)


def checkpoint_model_type(checkpoint: str, fallback: str) -> str:
    """The micro_sam model a trained checkpoint was finetuned from."""
    meta = Path(checkpoint).parent / META
    try:
        return json.loads(meta.read_text())["model_type"]
    except (OSError, ValueError, KeyError):
        return fallback


def _predictor_and_segmenter(model: str, checkpoint: str, tiled: bool):
    checkpoint = checkpoint or None
    mtime = Path(checkpoint).stat().st_mtime if checkpoint else None
    key = (model, checkpoint, mtime, tiled)
    if key not in _loaded:
        from .._microsam import sam1

        get_predictor_and_segmenter = sam1(
            "automatic_segmentation"
        ).get_predictor_and_segmenter

        _loaded.clear()
        progress(f"Loading micro_sam {checkpoint or model}")
        _loaded[key] = get_predictor_and_segmenter(
            model_type=model, checkpoint=checkpoint, is_tiled=tiled
        )
    return _loaded[key]


@op(env="pytorch")
def microsam(
    image: ImageOf[np.ndarray],
    model: str = "vit_b_lm",
    checkpoint: str = "",
    tile: bool = False,
    tile_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 64, "max": 4096, "step": 32}
    ] = 384,
    halo: Annotated[
        int, {"widget_type": "SpinBox", "min": 0, "max": 512}
    ] = 64,
    foreground_threshold: float = 0.5,
    center_distance_threshold: float = 0.5,
    boundary_distance_threshold: float = 0.5,
) -> LabelsOf[np.ndarray]:
    """Find every object in *image* with micro_sam.

    Args:
        image: YX, YXC or ZYX.
        model: micro_sam model, e.g. ``vit_b_lm``. With *checkpoint*, the one
            it was finetuned from, unless the checkpoint's ``skop_microsam.json``
            says otherwise.
        checkpoint: A ``best.pt`` from ``train_microsam``, or empty for the
            built-in *model*.
        tile: Embed in tiles, for images too large for one pass.
        tile_size, halo: Tile edge and overlap, in pixels.
        foreground_threshold, center_distance_threshold,
            boundary_distance_threshold: The instance decoder's settings.
            Ignored by models without one.

    Returns:
        A label image shaped like the image's spatial axes.
    """
    if checkpoint:
        model = checkpoint_model_type(checkpoint, model)
    predictor, segmenter = _predictor_and_segmenter(model, checkpoint, tile)

    from .._microsam import sam1

    automatic_instance_segmentation = sam1(
        "automatic_segmentation"
    ).automatic_instance_segmentation

    progress("Segmenting")
    labels = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=np.asarray(image),
        ndim=_ndim(image.shape),
        tile_shape=(tile_size, tile_size) if tile else None,
        halo=(halo, halo) if tile else None,
        verbose=False,
        foreground_threshold=foreground_threshold,
        center_distance_threshold=center_distance_threshold,
        boundary_distance_threshold=boundary_distance_threshold,
    )
    labels = np.asarray(labels)
    progress(f"Found {len(np.unique(labels)) - 1} objects")
    return labels.astype(np.uint32)


