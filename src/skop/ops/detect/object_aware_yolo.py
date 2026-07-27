"""Class-agnostic object detection with MobileSAMv2's object-aware YOLO.

A thin wrapper over ``segment-everything``, which vendors MobileSAMv2 from
https://github.com/ChaoningZhang/MobileSAM. The model is a YOLOv8 fine-tuned
to detect objects without classifying them -- built to produce box prompts for
SAM, which is exactly what it is used for here.

Wrapped rather than ported. The vendored tree is 28,000 lines across
efficientvit, tinyvit and a trimmed ultralytics 8.0.120 fork, with local
modifications; reimplementing it would mean maintaining a fork of a fork, and
an environment costs one pixi.toml. See docs/design/0007.

That vendored ultralytics is also why this op has an environment to itself:
it goes on sys.path at load time, so a maintained ultralytics installed
beside it would shadow or be shadowed by it. The FastSAM op lives in the
shared 'pytorch' environment for that reason.
"""

from __future__ import annotations

from typing import Annotated

from skop import boxes as _boxes
from skop import op, progress
from skop.assets import file_from_url
from skop.types import ImageData

from .._util import to_rgb
from ._result import Boxes

# The same weights segment_everything.weights_helper would fetch, from the
# HuggingFace mirror already listed in its WEIGHTS_URLS rather than from its
# Google Drive entry. Fetched here rather than through that module because
# weights_helper imports toolz and napari.utils.progress, neither of which
# segment-everything declares as a dependency -- so importing it means
# installing napari into a worker process to download a file.
_WEIGHTS = "https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/ObjectAwareModel.pt"


@op(env="segment-everything")
def object_aware_yolo(
    image: ImageData,
    conf: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.4,
    iou: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.9,
    max_det: int = 400,
    imgsz: int = 1024,
) -> Boxes:
    """Find every object in an image, without regard to what it is.

    Args:
        image: Image to detect in. 2-D or RGB; anything else is stretched to
            8-bit RGB first, which is what the model was trained on.
        conf: Confidence threshold. Lower finds more, and more spuriously.
        iou: NMS threshold. Two detections overlapping by more than this are
            treated as one object, so dense or touching objects need it high.
        max_det: Cap on how many objects to report.
        imgsz: Size the image is resized to for inference.

    Returns:
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.

    Note: the weights are 140 MB, downloaded into skop's asset cache on first
    use.
    """
    import torch
    from segment_everything.object_detectors.yolo_detector import YoloDetector

    weights = file_from_url(_WEIGHTS, "object_aware_yolo/ObjectAwareModel.pt")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    progress("Loading the object-aware model")
    detector = YoloDetector(
        str(weights),
        "ObjectAwareModelFromMobileSamV2",
        device=device,
    )

    progress("Detecting objects")
    # get_bounding_boxes() would do the same thing, but get_results keeps the
    # per-box confidences reachable for when they have somewhere to go.
    results = detector.get_results(
        to_rgb(image),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )

    # Its postprocess returns an empty list, not an empty result, when the
    # model finds nothing -- and prints "No object detected." on the way.
    if not results or results[0].boxes is None:
        return Boxes(_boxes.EMPTY.copy())

    found = results[0].boxes
    progress(f"Found {len(found)} objects")
    # found.conf holds a confidence per box. Not returned -- see _result.py.
    return Boxes(_boxes.from_xyxy(found.xyxy.cpu().numpy()))
