"""Point-prompted micro_sam, embedding once per image.

Interactive SAM is two costs of very different size: an embedding per image
(seconds, more for a volume) and a prompt per click (milliseconds). So these
are two ops sharing one module, and the module keeps what the expensive one
made. The worker is long-lived -- one per environment for the runner's
lifetime -- so a module-level dict outlives every call, which is all the
"session" this needs (scikit-ops design 0019, "Embeddings are the other
occupant").

Three levels of reuse, cheapest first:

1. In this worker's memory, keyed by the embedding path the caller passes.
   A few are kept, so flipping between two slices and back costs nothing.
2. On disk, as micro_sam's own zarr at that path. micro_sam stores a SHA1 of
   the image beside it and refuses a zarr made from different pixels, so a
   return visit -- even after napari restarts -- is a read, never a stale
   answer.
3. Recomputed, and saved to that path for next time.

Where embeddings live is the caller's project layout; the ops are only handed
a path. Uses micro_sam's public functions, not ``sam_annotator``, whose
``__init__`` imports napari.

Shares the 'pytorch' environment (0002) with the other micro_sam ops.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from opspec.op import op
from opspec.runtime import progress
from opspec.types import ImageOf, LabelsOf, PointsOf

#: How many images' embeddings stay in memory. Enough to flip between a few
#: slices instantly; each is the encoder's output, tens of MB per plane.
KEEP_EMBEDDINGS = 3

#: One predictor, for the model last asked for. Another model replaces it.
_predictor: dict = {}
#: (embedding_path, model_type) -> (image shape, image embeddings)
_embeddings: OrderedDict = OrderedDict()


def _get_predictor(model_type: str):
    if model_type not in _predictor:
        from .._microsam import sam1

        get_sam_model = sam1("util").get_sam_model

        _predictor.clear()
        progress(f"Loading micro_sam model {model_type}")
        _predictor[model_type] = get_sam_model(model_type=model_type)
    return _predictor[model_type]


def _get_embeddings(image, embedding_path: str, model_type: str, ndim: int):
    """This image's embeddings: from memory, from disk, or computed."""
    key = (embedding_path or None, model_type)
    held = _embeddings.get(key)
    if key[0] is not None and held is not None and held[0] == image.shape:
        _embeddings.move_to_end(key)
        return held[1]

    from .._microsam import sam1

    precompute_image_embeddings = sam1("util").precompute_image_embeddings

    predictor = _get_predictor(model_type)
    progress("Computing or loading image embeddings")
    try:
        embeddings = precompute_image_embeddings(
            predictor,
            image,
            save_path=key[0],
            ndim=ndim,
            verbose=False,
        )
    except RuntimeError as e:
        # A zarr at that path made from other pixels or another model. It is
        # a cache, but someone else's files: compute without it rather than
        # delete it.
        if "Embeddings file" not in str(e):
            raise
        progress(f"Ignoring a mismatched embedding file: {e}")
        embeddings = precompute_image_embeddings(
            predictor, image, save_path=None, ndim=ndim, verbose=False
        )

    if key[0] is not None:
        _embeddings[key] = (image.shape, embeddings)
        while len(_embeddings) > KEEP_EMBEDDINGS:
            _embeddings.popitem(last=False)
    return embeddings


@op(env="pytorch")
def microsam_embed(
    image: ImageOf[np.ndarray],
    embedding_path: str = "",
    model_type: str = "vit_b_lm",
    ndim: int = 2,
) -> int:
    """Make *image*'s micro_sam embeddings ready for prompting.

    Args:
        image: 2-D (YX or YXC) or 3-D (ZYX) image to prompt later.
        embedding_path: Zarr to load embeddings from, or save them to. Empty
            keeps them in memory only, for this one call.
        model_type: micro_sam checkpoint, e.g. ``vit_b_lm``.
        ndim: 2 for an image, 3 for a volume.

    Returns:
        The number of images now held in memory.
    """
    _get_embeddings(image, embedding_path, model_type, ndim)
    return len(_embeddings)


@op(env="pytorch")
def microsam_points(
    image: ImageOf[np.ndarray],
    points: PointsOf[np.ndarray],
    point_labels: np.ndarray,
    embedding_path: str = "",
    model_type: str = "vit_b_lm",
    ndim: int = 2,
    volume: bool = True,
    iou_threshold: float = 0.7,
    box_extension: float = 0.5,
) -> LabelsOf[np.ndarray]:
    """Segment one object from point prompts.

    Args:
        image: As given to ``microsam_embed``.
        points: (N, 2) as [y, x], or (N, 3) as [z, y, x] for a volume -- all
            on the same z, the plane being clicked.
        point_labels: (N,) 1 for a positive point, 0 for a negative one.
        embedding_path, model_type, ndim: As given to ``microsam_embed``.
        volume: For a volume, carry the object through z from the clicked
            plane; otherwise segment that plane only.
        iou_threshold, box_extension: micro_sam's settings for carrying an
            object from one plane to the next.

    Returns:
        uint8 mask shaped like the image's spatial axes: 1 for the object.
    """
    points = np.asarray(points, dtype=float).reshape(len(points), -1)
    labels = np.asarray(point_labels, dtype=int).reshape(-1)
    spatial = image.shape[:ndim]
    if len(points) == 0:
        return np.zeros(spatial, dtype=np.uint8)

    from .._microsam import sam1

    segment_from_points = sam1("prompt_based_segmentation").segment_from_points

    predictor = _get_predictor(model_type)
    embeddings = _get_embeddings(image, embedding_path, model_type, ndim)

    if ndim == 3:
        z = int(round(points[0][0]))
        yx = points[:, 1:3]
    else:
        z = None
        yx = points[:, 0:2]

    plane = np.squeeze(
        segment_from_points(
            predictor, yx, labels, image_embeddings=embeddings, i=z
        )
    ).astype(np.uint8)

    if ndim != 3:
        return plane

    seg = np.zeros(spatial, dtype=np.uint8)
    seg[z] = plane
    if not volume or not plane.any():
        return seg

    segment_mask_in_volume = sam1(
        "multi_dimensional_segmentation"
    ).segment_mask_in_volume

    progress("Carrying the object through z")
    seg, _ = segment_mask_in_volume(
        seg,
        predictor,
        embeddings,
        np.array([z]),
        stop_lower=False,
        stop_upper=False,
        iou_threshold=iou_threshold,
        projection="single_point",
        box_extension=box_extension,
    )
    return np.asarray(seg).astype(np.uint8)
