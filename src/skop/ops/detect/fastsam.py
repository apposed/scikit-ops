"""Class-agnostic object detection with FastSAM.

FastSAM is a YOLOv8 segmentation model trained on 2% of SA-1B. Ultralytics
describes it as recognising and segmenting "all objects as the same class" --
it has one class, called object, so its confidence score means "is this a
thing" rather than "is this one of 80 COCO categories".

That is the distinction that matters here. A COCO-pretrained detector
localises coins or cells perfectly well -- box regression is class-agnostic --
and then scores every class near zero and throws them all away below the
confidence threshold. A single-class model keeps them.

Only the boxes are returned. FastSAM also produces masks, from a prototype
branch cropped by these same boxes; a mask detector op built on that is
possible later (see docs/design/0007) but is not this op's job.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from skop import boxes as _boxes
from skop import op, progress
from skop.assets import file_from_url
from skop.types import ImageData

from .._util import to_rgb
from ._result import Boxes

_WEIGHTS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"


class PretrainedModel(Enum):
    """FastSAM's two published checkpoints."""

    small = "FastSAM-s.pt"
    large = "FastSAM-x.pt"


@op(env="pytorch")
def fastsam(
    image: ImageData,
    model: PretrainedModel = PretrainedModel.small,
    conf: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.4,
    iou: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.9,
    max_det: int = 300,
    imgsz: int = 1024,
) -> Boxes:
    """Find every object in an image, without regard to what it is.

    Args:
        image: Image to detect in. 2-D or RGB; anything else is stretched to
            8-bit RGB first, which is what the model was trained on.
        model: Which checkpoint to use. The small one is 24 MB, the large
            145 MB.
        conf: Confidence threshold. Lower finds more, and more spuriously.
        iou: NMS threshold. Two detections overlapping by more than this are
            treated as one object, so dense or touching objects need it high.
        max_det: Cap on how many objects to report.
        imgsz: Size the image is resized to for inference.

    Returns:
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.

    Note: the weights are downloaded into skop's asset cache on first use.
    """
    from ultralytics import FastSAM

    weights = file_from_url(_WEIGHTS.format(name=model.value), f"fastsam/{model.value}")

    progress(f"Loading {model.value}")
    net = FastSAM(str(weights))

    progress("Detecting objects")
    results = net.predict(
        to_rgb(image),
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        retina_masks=False,
        verbose=False,
    )

    # An image with nothing in it gives back an empty result list, not a
    # result holding zero boxes.
    if not results or results[0].boxes is None:
        return Boxes(_boxes.EMPTY.copy())

    found = results[0].boxes
    progress(f"Found {len(found)} objects")
    # found.conf holds a confidence per box. Not returned -- see _result.py.
    return Boxes(_boxes.from_xyxy(found.xyxy.cpu().numpy()))
