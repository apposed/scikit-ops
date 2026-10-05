"""YOLO on overlapping square tiles, with a measured GPU batch size."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from skop import Axes, cancel_requested, op, progress
from skop.types import ImageData

from .._util import to_rgb
from ._result import Detections


@op(env="pytorch")
def yolo(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    weights: Path,
    object_size: float | None = None,
    overlap: Annotated[float, {"min": 0.0, "max": 0.95}] = 0.15,
    gpu_fraction: Annotated[float, {"min": 0.01, "max": 1.0}] = 0.9,
    conf: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.25,
    iou: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
    max_det: int = 300,
) -> Detections:
    """Detect objects in a large image with a YOLO .pt checkpoint.

    Args:
        image: A 2-D image or trailing RGB(A) channels. Only the current
            tiles are converted to RGB and resized; the image stays on CPU.
        weights: Local YOLO .pt file. Its saved training size sets the
            inference size; it is rounded up to the model's stride.
        object_size: Typical object area in pixels squared. None runs the
            whole image once. An area outside 0.1%-50% of the image selects
            square tiles of side ceil(sqrt(1.5 * object_size / 0.001)).
        overlap: Fraction of a tile shared with its neighbour. The final
            tile along each axis is shifted back to meet the image edge.
        gpu_fraction: Fraction of free memory after loading the model to
            budget for batches. CUDA uses a single-tile allocation peak;
            MPS samples driver memory against its recommended working set
            and available system RAM. An out-of-memory batch is halved.
        conf: Minimum detection confidence.
        iou: Per-tile NMS IoU threshold. Across tiles this is the intersection
            over smaller box threshold, which also removes cropped duplicates.
        max_det: Maximum detections per tile; there is no whole-image cap.

    Returns:
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.
        confidences: One confidence per box, highest first.
        classes: One integer class ID per box, in the same order.

    Square tiles only need resizing. If an image axis is shorter than the
    tile, its unused region is padded at inference resolution, preserving
    aspect ratio without allocating a large padded image. The input itself
    must fit in host memory; this is not an out-of-core runner.
    """
    import gc
    import math
    import threading

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not Path(weights).is_file() or Path(weights).suffix.lower() != ".pt":
        raise ValueError("weights must name an existing YOLO .pt file")
    if not 0 <= overlap < 1 or not 0 < gpu_fraction <= 1:
        raise ValueError("overlap must be in [0, 1), gpu_fraction in (0, 1]")
    if object_size is not None and (not math.isfinite(object_size) or object_size <= 0):
        raise ValueError("object_size must be a positive area or None")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError("expected a 2-D image or trailing RGB(A) channels")

    height, width = image.shape[:2]
    if not height or not width:
        raise ValueError("image dimensions must be nonzero")
    side = max(height, width)
    if object_size is not None and not 0.001 <= object_size / (height * width) <= 0.5:
        side = math.ceil(math.sqrt(1.5 * object_size / 0.001))
    step = max(1, int(side * (1 - overlap)))
    ys = list(range(0, max(0, height - side), step)) + [max(0, height - side)]
    xs = list(range(0, max(0, width - side), step)) + [max(0, width - side)]
    origins = [(y, x) for y in ys for x in xs]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    gpu = getattr(torch, device.split(":")[0]) if device != "cpu" else None
    progress(f"Loading {Path(weights).name} on {device}")
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("weights must be a YOLO object-detection checkpoint")
    imgsz = model.model.args["imgsz"]
    imgsz = max(imgsz) if isinstance(imgsz, (list, tuple)) else imgsz
    stride = int(model.model.stride.max())
    imgsz = math.ceil(imgsz / stride) * stride

    def infer(coords):
        inputs = np.full((len(coords), 3, imgsz, imgsz), 114, dtype=np.uint8)
        scales = []
        for slot, (y, x) in enumerate(coords):
            crop = to_rgb(image[y : y + side, x : x + side])
            h, w = crop.shape[:2]
            rh, rw = max(1, round(h * imgsz / side)), max(1, round(w * imgsz / side))
            inputs[slot, :, :rh, :rw] = cv2.resize(crop, (rw, rh)).transpose(2, 0, 1)
            scales.append((rw / w, rh / h))
        inputs = torch.from_numpy(inputs).float().div_(255)
        results = model.predict(
            inputs,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch=len(coords),
            verbose=False,
            save=False,
        )
        found = []
        for result, (y, x), (sx, sy) in zip(results, coords, scales):
            boxes = result.boxes.data[:, :6].cpu().numpy().copy()
            # Prefer an overlapping tile's full box to a cut-off edge box.
            partial = (
                ((boxes[:, 0] <= 1) & (x > 0))
                | ((boxes[:, 1] <= 1) & (y > 0))
                | ((boxes[:, 2] >= min(side, width - x) * sx - 1) & (x + side < width))
                | (
                    (boxes[:, 3] >= min(side, height - y) * sy - 1)
                    & (y + side < height)
                )
            )
            boxes = np.column_stack(
                (boxes, partial, np.full(len(boxes), y * width + x))
            )
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]] / sx + x, x, min(x + side, width)
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / sy + y, y, min(y + side, height)
            )
            found.append(
                boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            )
        # The predictor holds its last results, including GPU tensors.
        model.predictor.results = None
        return found

    try:
        model.to(device)
        if gpu is not None:
            gpu.synchronize()
            gpu.empty_cache()
        if device.startswith("cuda"):
            free, _ = gpu.mem_get_info()
            baseline = gpu.memory_allocated()
            gpu.reset_peak_memory_stats()
        elif device == "mps":
            import psutil

            baseline = gpu.driver_allocated_memory()
            free = min(
                gpu.recommended_max_memory() - baseline,
                psutil.virtual_memory().available,
            )

        peak = [baseline if gpu is not None else 0]
        stop = threading.Event()

        def sample_memory():
            while not stop.wait(0.001):
                peak[0] = max(peak[0], gpu.driver_allocated_memory())

        monitor = threading.Thread(target=sample_memory) if device == "mps" else None
        if monitor is not None:
            monitor.start()
        try:
            if cancel_requested():
                raise RuntimeError("YOLO cancelled")
            progress(f"Profiling one of {len(origins)} tiles ({side} px)")
            found = infer(origins[:1])
            if gpu is not None:
                gpu.synchronize()
            if device == "mps":
                peak[0] = max(peak[0], gpu.driver_allocated_memory())
        finally:
            stop.set()
            if monitor is not None:
                monitor.join()

        batch_size = 1
        if gpu is not None and len(origins) > 1:
            if device.startswith("cuda"):
                peak[0] = gpu.max_memory_allocated()
            per_tile = max(peak[0] - baseline, 3 * imgsz * imgsz * 4)
            batch_size = min(
                len(origins) - 1, max(1, int(gpu_fraction * free / per_tile))
            )
        progress(
            f"Detecting {len(origins)} tiles, batch size {batch_size}", 1, len(origins)
        )

        done = 1
        while done < len(origins):
            if cancel_requested():
                raise RuntimeError("YOLO cancelled")
            coords = origins[done : done + batch_size]
            try:
                batch = infer(coords)
            except RuntimeError as exc:
                if (
                    gpu is None
                    or len(coords) == 1
                    or "out of memory" not in str(exc).lower()
                ):
                    raise
                batch_size = max(1, len(coords) // 2)
                progress(f"Retrying with batch size {batch_size}")
            else:
                found.extend(batch)
                done += len(coords)
                progress(f"Detected {done} of {len(origins)} tiles", done, len(origins))
                continue
            # The exception's traceback has now released the failed inputs.
            model.predictor.results = None
            gpu.empty_cache()

        detections = np.concatenate(found)
        areas = (detections[:, 2] - detections[:, 0]) * (
            detections[:, 3] - detections[:, 1]
        )
        order = np.lexsort((-detections[:, 4], detections[:, 6]))
        keep = []
        while order.size:
            first, rest = order[0], order[1:]
            keep.append(first)
            intersection = np.maximum(
                0,
                np.minimum(detections[first, 2:4], detections[rest, 2:4])
                - np.maximum(detections[first, :2], detections[rest, :2]),
            ).prod(axis=1)
            duplicate = (
                (detections[first, 5] == detections[rest, 5])
                & (detections[first, 7] != detections[rest, 7])
                & (intersection > iou * np.minimum(areas[first], areas[rest]))
            )
            order = rest[~duplicate]
        detections = detections[keep]
        detections = detections[np.argsort(-detections[:, 4], kind="stable")]
        progress(f"Found {len(detections)} objects", len(origins), len(origins))
        return Detections(
            detections[:, [1, 0, 3, 2]],
            detections[:, 4].tolist(),
            detections[:, 5].astype(int).tolist(),
        )
    finally:
        model = None
        gc.collect()
        if gpu is not None:
            gpu.empty_cache()
