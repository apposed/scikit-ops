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
    merge_threshold: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
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
            budget for batches after warm-up. Actual batch peaks adjust the
            batch size up or down; out-of-memory batches are halved. MPS
            samples driver memory against its recommended working set.
        conf: Minimum detection confidence.
        iou: Per-tile NMS IoU threshold.
        max_det: Maximum detections per tile; there is no whole-image cap.
        merge_threshold: IoS threshold for class-aware GreedyNMM, matching
            SAHI 0.12.8. Matching boxes, including within one tile, become an
            enclosing box with the maximum confidence. Equal overlap matches.

    Returns:
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.
        confidences: One confidence per box. Outputs follow SAHI's class order,
            then the original keeper order within each class.
        classes: One integer class ID per box, in the same order.

    Square tiles only need resizing. If an image axis is shorter than the
    tile, its unused region is padded at inference resolution, preserving
    aspect ratio without allocating a large padded image. The input itself
    must fit in host memory; this is not an out-of-core runner.
    """
    import contextvars
    import gc
    import math
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO

    # Ultralytics moved NMS out of ops during the 8.3 series.
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    if not Path(weights).is_file() or Path(weights).suffix.lower() != ".pt":
        raise ValueError("weights must name an existing YOLO .pt file")
    if not 0 <= overlap < 1 or not 0 < gpu_fraction <= 1:
        raise ValueError("overlap must be in [0, 1), gpu_fraction in (0, 1]")
    if object_size is not None and (not math.isfinite(object_size) or object_size <= 0):
        raise ValueError("object_size must be a positive area or None")
    if not 0 <= merge_threshold <= 1:
        raise ValueError("merge_threshold must be in [0, 1]")
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

    def prepare(coords):
        inputs = torch.empty(
            (len(coords), 3, imgsz, imgsz),
            dtype=torch.float32,
            pin_memory=device.startswith("cuda"),
        )
        pixels = inputs.numpy()
        pixels.fill(114 / 255)
        scales = []
        for slot, (y, x) in enumerate(coords):
            if cancel_requested():
                raise RuntimeError("YOLO cancelled")
            crop = to_rgb(image[y : y + side, x : x + side])
            h, w = crop.shape[:2]
            rh, rw = max(1, round(h * imgsz / side)), max(1, round(w * imgsz / side))
            np.multiply(
                cv2.resize(crop, (rw, rh)).transpose(2, 0, 1),
                1 / 255,
                out=pixels[slot, :, :rh, :rw],
            )
            scales.append((rw / w, rh / h))
        return inputs, coords, scales

    def infer(prepared, profile=False):
        inputs, coords, scales = prepared
        peak = [baseline]
        stop = threading.Event()

        def sample_memory():
            while not stop.wait(0.001):
                peak[0] = max(peak[0], gpu.driver_allocated_memory())

        monitor = None
        if profile:
            if device.startswith("cuda"):
                gpu.reset_peak_memory_stats()
            else:
                peak[0] = gpu.driver_allocated_memory()
                monitor = threading.Thread(target=sample_memory)
                monitor.start()
        try:
            with torch.inference_mode():
                outputs = non_max_suppression(
                    model.model(
                        inputs.to(device, non_blocking=device.startswith("cuda"))
                    ),
                    conf_thres=conf,
                    iou_thres=iou,
                    max_det=max_det,
                    nc=len(model.names),
                )
                counts = [len(boxes) for boxes in outputs]
                # Copy detections once; skip predict()'s image/Results conversion.
                rows = torch.cat([boxes[:, :6] for boxes in outputs]).cpu().numpy()
            if profile:
                gpu.synchronize()
                peak[0] = (
                    gpu.max_memory_reserved()
                    if device.startswith("cuda")
                    else max(peak[0], gpu.driver_allocated_memory())
                )
        finally:
            stop.set()
            if monitor is not None:
                monitor.join()

        found = []
        for boxes, (y, x), (sx, sy) in zip(
            np.split(rows, np.cumsum(counts)[:-1]),
            coords,
            scales,
        ):
            boxes = boxes.astype(np.float64)
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]] / sx + x, x, min(x + side, width)
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / sy + y, y, min(y + side, height)
            )
            found.append(
                boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            )
        return found, max(peak[0] - baseline, inputs.numel() * inputs.element_size())

    baseline = 0
    try:
        model.to(device)
        model.model.fuse(verbose=False).eval().float()
        progress(f"Warming model on the first of {len(origins)} tiles ({side} px)")
        found, _ = infer(prepare(origins[:1]))
        batch_size, best_size, batch_limit = 1, 1, len(origins)
        if gpu is not None and len(origins) > 1:
            # Keep setup costs resident, but return unused activation caches.
            gpu.synchronize()
            gpu.empty_cache()
            if device.startswith("cuda"):
                baseline = gpu.memory_reserved()
                free, _ = gpu.mem_get_info()
            else:
                baseline = gpu.driver_allocated_memory()
                free = min(
                    gpu.recommended_max_memory() - baseline,
                    psutil.virtual_memory().available,
                )
            budget = gpu_fraction * free
            # Current and prefetched batches coexist in host memory.
            batch_limit = min(
                batch_limit,
                max(
                    1,
                    int(
                        0.9
                        * psutil.virtual_memory().available
                        / (2 * 3 * imgsz * imgsz * 4)
                    ),
                ),
            )

        done = 1
        with ThreadPoolExecutor(max_workers=1) as loader:

            def submit(start, count):
                return loader.submit(
                    contextvars.copy_context().run,
                    prepare,
                    origins[start : start + count],
                )

            current = submit(done, batch_size) if done < len(origins) else None
            while current is not None:
                if cancel_requested():
                    current.cancel()
                    raise RuntimeError("YOLO cancelled")
                prepared = current.result()
                if len(prepared[1]) > batch_size:
                    prepared = tuple(part[:batch_size] for part in prepared)
                count = len(prepared[1])
                next_start = done + count
                pending = (
                    submit(next_start, batch_size)
                    if next_start < len(origins)
                    else None
                )
                try:
                    batch, used = infer(prepared, profile=gpu is not None)
                except RuntimeError as exc:
                    if (
                        gpu is None
                        or count == 1
                        or "out of memory" not in str(exc).lower()
                    ):
                        raise
                    if pending is not None:
                        pending.cancel()
                    batch_limit = min(batch_limit, count - 1)
                    best_size = min(best_size, batch_limit)
                    batch_size = max(1, count // 2)
                    progress(f"Retrying with batch size {batch_size}")
                else:
                    found.extend(batch)
                    done += count
                    if gpu is not None:
                        # Feedback from real, warmed batches corrects the estimate.
                        if used <= budget:
                            best_size = max(best_size, count)
                        else:
                            batch_limit = min(batch_limit, max(1, count - 1))
                            best_size = min(best_size, batch_limit)
                        estimate = max(1, int(count * budget / used))
                        # Small-batch allocator plans extrapolate poorly.
                        if count < 32:
                            estimate = min(estimate, max(8, count * 2))
                        batch_size = min(batch_limit, estimate)
                        if estimate > batch_limit and best_size < batch_limit:
                            batch_size = (best_size + batch_limit + 1) // 2
                        if batch_size != count:
                            gpu.empty_cache()
                        progress(
                            f"Detected {done} of {len(origins)} tiles; "
                            f"batch {count}, peak {used / 2**20:.0f} MiB, next {batch_size}",
                            done,
                            len(origins),
                        )
                    else:
                        progress(
                            f"Detected {done} of {len(origins)} tiles",
                            done,
                            len(origins),
                        )
                    del prepared
                    current = pending
                    continue
                # Drop failed-batch tracebacks before releasing the GPU cache.
                del prepared
                gpu.empty_cache()
                current = submit(done, batch_size)

        detections = _greedy_nmm(np.concatenate(found), merge_threshold)
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


def _greedy_nmm(detections, threshold=0.5):
    """SAHI 0.12.8 GreedyNMM, with a multiscale grid for nearby candidates."""
    import math

    import numpy as np

    original = np.asarray(detections, dtype=np.float64)[:, :6]
    matching = original.astype(np.float32)
    areas = (matching[:, 2] - matching[:, 0]) * (matching[:, 3] - matching[:, 1])
    merged = []
    active = np.ones(len(original), dtype=bool)
    rank = np.empty(len(original), dtype=np.intp)

    for category in np.unique(matching[:, 5]):
        indices = np.flatnonzero(matching[:, 5] == category)
        rows = matching[indices]
        order = indices[
            np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0], -rows[:, 4]))
        ]
        rank[order] = np.arange(len(order))
        levels, positions = {}, {}

        if threshold > 0:
            for index in order:
                x1, y1, x2, y2 = map(float, matching[index, :4])
                level = math.ceil(math.log2(max(1, x2 - x1, y2 - y1)))
                size = 2.0**level
                cell = (
                    math.floor((x1 + x2) / (2 * size)),
                    math.floor((y1 + y2) / (2 * size)),
                )
                levels.setdefault(level, {}).setdefault(cell, set()).add(int(index))
                positions[int(index)] = (level, cell)

        def remove(index, positions=positions, levels=levels):
            level, cell = positions[index]
            members = levels[level][cell]
            members.remove(index)
            if not members:
                del levels[level][cell]

        def neighbours(box, levels=levels):
            candidates = []
            x1, y1, x2, y2 = map(float, box)
            for level, cells in levels.items():
                size = 2.0**level
                left, top = math.floor(x1 / size - 0.5), math.floor(y1 / size - 0.5)
                right, bottom = math.floor(x2 / size + 0.5), math.floor(y2 / size + 0.5)
                if (right - left + 1) * (bottom - top + 1) < len(cells):
                    for y in range(top, bottom + 1):
                        for x in range(left, right + 1):
                            candidates.extend(cells.get((x, y), ()))
                else:
                    for (x, y), members in cells.items():
                        if left <= x <= right and top <= y <= bottom:
                            candidates.extend(members)
            return np.asarray(candidates, dtype=np.intp)

        for index in order:
            if cancel_requested():
                raise RuntimeError("YOLO cancelled")
            if not active[index]:
                continue

            if threshold <= 0:
                candidates = order[1:]
            else:
                remove(int(index))
                candidates = neighbours(matching[index, :4])
                if candidates.size:
                    boxes = matching[candidates, :4]
                    intersection = np.maximum(
                        0,
                        np.minimum(matching[index, 2:4], boxes[:, 2:4])
                        - np.maximum(matching[index, :2], boxes[:, :2]),
                    ).prod(axis=1)
                    denominator = np.minimum(areas[index], areas[candidates])
                    ios = np.divide(
                        intersection,
                        denominator,
                        out=np.zeros_like(intersection),
                        where=denominator > 0,
                    )
                    candidates = candidates[ios >= threshold]
                    candidates = candidates[np.argsort(rank[candidates])]
                    for candidate in candidates:
                        remove(int(candidate))

            # Claim against original boxes before growing the keeper. GreedyNMM
            # does not acquire new matches from the enlarged bounding box.
            active[candidates] = False
            keeper = original[index].copy()
            for candidate in candidates:
                if cancel_requested():
                    raise RuntimeError("YOLO cancelled")
                box = original[candidate]
                intersection = np.maximum(
                    0,
                    np.minimum(keeper[2:4], box[2:4]) - np.maximum(keeper[:2], box[:2]),
                ).prod()
                denominator = min(
                    (keeper[2] - keeper[0]) * (keeper[3] - keeper[1]),
                    (box[2] - box[0]) * (box[3] - box[1]),
                )
                if denominator > 0 and intersection / denominator >= threshold:
                    keeper[:2] = np.minimum(keeper[:2], box[:2])
                    keeper[2:4] = np.maximum(keeper[2:4], box[2:4])
                    if box[4] >= keeper[4]:
                        keeper[4:6] = box[4:6]
            merged.append(keeper)

    return np.asarray(merged, dtype=np.float64).reshape(-1, 6)
