"""Class-aware GreedyNMM with SAHI 0.12.8's grouping and merge semantics.

Reference: https://github.com/obss/sahi/tree/0.12.8/sahi/postprocess
Matching uses float32, followed by sequential unions in the original precision.
A multiscale grid indexes original box centres, one entry per box. Queries
expand by half a level's cell size, which bounds every box at that level.
"""

from __future__ import annotations

import math

import numpy as np

from skop import cancel_requested


def greedy_nmm(detections: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Merge (N, 6) xyxy/confidence/class rows, in SAHI's class/keeper order."""
    original = np.asarray(detections, dtype=np.float64)[:, :6]
    matching = original.astype(np.float32)
    areas = (matching[:, 2] - matching[:, 0]) * (matching[:, 3] - matching[:, 1])
    merged = []

    for category in np.unique(matching[:, 5]):
        indices = np.flatnonzero(matching[:, 5] == category)
        rows = matching[indices]
        order = indices[
            np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0], -rows[:, 4]))
        ]
        rank = dict(zip(order.tolist(), range(len(order))))
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

        claimed = set()
        for index in order:
            if cancel_requested():
                raise RuntimeError("YOLO cancelled")
            if index in claimed:
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
                    candidates = np.asarray(
                        sorted(candidates, key=rank.__getitem__), dtype=np.intp
                    )
                    for candidate in candidates:
                        remove(int(candidate))

            # Claim against original boxes before growing the keeper. GreedyNMM
            # does not acquire new matches from the enlarged bounding box.
            claimed.update(candidates.tolist())
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
