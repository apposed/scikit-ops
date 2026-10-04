"""YOLO detection with any ultralytics weights, ported from JDLL.

JDLL (https://github.com/bioimage-io/JDLL) runs YOLO from Java by building
Python as strings in ``model/special/yolo/Yolo.java`` and sending it to an
Appose worker. This is that Python, as an op. The readable copy it came from
is docs/spec/jdll-yolo-infer.py.

Kept from JDLL: the device choice, the hand-written letterbox, and handing
ultralytics a tensor, which skips its own preprocessing. Dropped: the shared
memory, the axis-letter reordering, and the UUID-suffixed names, all of which
the runner already does or makes unnecessary.

Unlike ``fastsam`` and ``object_aware_yolo`` this is not class-agnostic: it
runs whatever model it is given, trained for whatever classes. The classes
are dropped for now, as everywhere in this package -- see _result.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from skop import boxes as _boxes
from skop import op, progress
from skop.types import ImageData

from .._util import to_rgb
from ._result import Boxes


@op(env="pytorch")
def jdll_yolo(
    image: ImageData,
    weights: Path,
    conf: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.25,
    iou: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = 0.7,
    imgsz: int = 640,
) -> Boxes:
    """Find objects with a trained YOLO model.

    Args:
        image: Image to detect in. 2-D or RGB; anything else is stretched to
            8-bit RGB first.
        weights: A YOLO ``.pt`` file, such as the ``best.pt`` training writes.
        conf: Confidence threshold. Lower finds more, and more spuriously.
        iou: NMS threshold. Two detections overlapping by more than this are
            treated as one object.
        imgsz: Size the longest side is resized to. JDLL fixes this at 640.

    Returns:
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # An empty file picker sends Path('.'), which YOLO accepts and only
    # rejects deep inside predict.
    if not Path(weights).is_file():
        raise FileNotFoundError(f"weights must be a YOLO .pt file, got {str(weights)!r}")

    progress(f"Loading {Path(weights).name}")
    model = YOLO(str(weights))

    # (Y, X, 3) uint8 -> (1, 3, Y, X) float, still 0-255. Ultralytics sees
    # values above 1 and divides by 255 itself, with a warning.
    rgb = to_rgb(image)
    x = torch.as_tensor(
        np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), dtype=torch.float32
    )
    x = x.to(device)

    # JDLL's letterbox: longest side to imgsz, then pad each side to a
    # multiple of 32, centred, with ultralytics' grey.
    _, _, height, width = x.shape
    scale = imgsz / max(height, width)
    new_h, new_w = round(height * scale), round(width * scale)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h, pad_w = (-new_h) % 32, (-new_w) % 32
    top, left = pad_h // 2, pad_w // 2
    x = F.pad(x, (left, pad_w - left, top, pad_h - top), value=114.0)

    progress("Detecting objects")
    results = model(x.contiguous(), conf=conf, iou=iou, device=device, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return Boxes(_boxes.EMPTY.copy())

    found = results[0].boxes
    progress(f"Found {len(found)} objects")

    # Undo the letterbox: shift off the padding, scale back, clip to the image.
    xyxy = found.xyxy.cpu().numpy()
    xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - left) / scale, 0, width)
    xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - top) / scale, 0, height)
    return Boxes(_boxes.from_xyxy(xyxy))
