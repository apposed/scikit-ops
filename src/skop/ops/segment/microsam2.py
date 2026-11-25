"""micro_sam 2 automatic instance segmentation: every object, no prompts.

The SAM2 twin of ``skop.ops.segment.microsam``. Its knobs are not micro_sam
1's, hence an op of its own. Two engines, picked by the model:

* A finetuned model (``hvit_*_cells``, ``hvit_s_em_*``) has a decoder, and
  runs micro_sam's "AIS": the decoder predicts foreground and boundaries, and
  the AIS knobs turn those into objects. A knob left as None takes the
  model's tuned default (micro_sam's ``DEFAULT_POSTPROCESSING``).
* A base SAM2 model (``hvit_t/s/b/l``, Meta's weights) has none, and runs
  "AMG": a grid of point prompts, a mask per point, the good ones kept. It
  segments anything object-like, including what the finetuning data left
  unlabelled (e.g. the red blood cells of NeurIPS 2022).

Shares the 'pytorch' environment (0002), whose micro_sam 2 carries both.
"""

from __future__ import annotations

import numpy as np
from opspec.op import op
from opspec.runtime import progress
from opspec.types import ImageOf, LabelsOf

from .microsam import _ndim

#: Finetuned micro_sam 2 models with a decoder, by what they were trained on.
MODELS = (
    "hvit_t_cells",
    "hvit_s_cells",
    "hvit_b_cells",
    "hvit_l_cells",
    "hvit_s_em_mito",
    "hvit_s_em_nuclei",
)

#: SAM2 itself, no decoder: these run AMG.
BASE_MODELS = ("hvit_t", "hvit_s", "hvit_b", "hvit_l")

#: One predictor and segmenter, for the settings last asked for.
_loaded: dict = {}


def _predictor_and_segmenter(model: str, mode: str, **amg):
    key = (model, mode, tuple(sorted(amg.items())))
    if key not in _loaded:
        from micro_sam.v2.automatic_segmentation import (
            get_predictor_and_segmenter,
        )

        _loaded.clear()
        progress(f"Loading micro_sam 2 {model} ({mode})")
        _loaded[key] = get_predictor_and_segmenter(
            model_type=model, segmentation_mode=mode, **amg
        )
    return _loaded[key]


@op(env="pytorch")
def microsam2(
    image: ImageOf[np.ndarray],
    model: str = "hvit_t_cells",
    foreground_threshold: float | None = None,
    density_threshold: float | None = None,
    min_size: int | None = None,
    foreground_weight: float | None = None,
    sigma: float | None = None,
    n_iter: int | None = None,
    dt: float | None = None,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.85,
    min_object_size: int = 0,
    max_object_size: int | None = None,
    tiled: bool | None = None,
) -> LabelsOf[np.ndarray]:
    """Find every object in *image* with micro_sam 2.

    Args:
        image: YX, YXC or ZYX.
        model: A finetuned model (``hvit_t_cells`` ...), run with its decoder,
            or a base SAM2 model (``hvit_b`` ...), run as AMG.
        foreground_threshold: Decoder models. Lower finds more, and
            fainter, objects.
        density_threshold: Decoder models. Seeds need this much flow
            converging on them; lower splits more.
        min_size: Decoder models. Smallest object kept, in pixels.
        foreground_weight: Decoder models. How much the foreground map
            counts against the boundary map when splitting touching objects.
        sigma, n_iter, dt: Decoder models. Smoothing, steps and step size of
            the flow integration that groups pixels into objects.
        pred_iou_thresh: Base models. Masks SAM2 scores lower are dropped.
        stability_score_thresh: Base models. Masks that change shape with
            the cut-off are dropped.
        min_object_size, max_object_size: Base models. Object area limits,
            in pixels; None for no upper limit.
        tiled: Split the image into overlapping tiles. None: micro_sam's
            choice (tiles above 768 px) for decoder models, untiled for base
            models, where tile borders cut objects.

    Returns:
        A label image shaped like the image's spatial axes.
    """
    from micro_sam.v2.automatic_segmentation import (
        automatic_instance_segmentation,
    )

    if model in BASE_MODELS:
        predictor, segmenter = _predictor_and_segmenter(
            model,
            "amg",
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        knobs = {
            "min_object_size": min_object_size,
            "max_object_size": max_object_size,
        }
        tiled = bool(tiled)
    else:
        predictor, segmenter = _predictor_and_segmenter(model, "ais")
        knobs = {
            "foreground_threshold": foreground_threshold,
            "density_threshold": density_threshold,
            "min_size": min_size,
            "foreground_weight": foreground_weight,
            "sigma": sigma,
            "n_iter": n_iter,
            "dt": dt,
        }
    if tiled is False:
        knobs.update(tile_shape=(0, 0), halo=(0, 0))

    progress("Segmenting")
    labels = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=np.asarray(image),
        ndim=_ndim(image.shape),
        verbose=False,
        **{k: v for k, v in knobs.items() if v is not None},
    )
    labels = np.asarray(labels)
    progress(f"Found {len(np.unique(labels)) - 1} objects")
    return labels.astype(np.uint32)
