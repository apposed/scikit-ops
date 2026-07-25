"""Toy ops, used to exercise opkit without dragging in a deep learning stack.

Note the shape of an op module: imports at the top are limited to what the
signatures need, so this module loads in a minimal environment. Anything an
op needs only in order to *run* is imported inside the function body.
"""

from __future__ import annotations

import time
from typing import Annotated, NamedTuple

import numpy as np

from opkit import Out, cancel_requested, op, progress


@op(env="minimal")
def add(a: int, b: int) -> int:
    """Add two numbers. The smallest op there is."""
    return a + b


class ScaleResult(NamedTuple):
    scaled: np.ndarray
    total: float


@op(env="minimal")
def scale(
    image: np.ndarray,
    factor: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 10.0, "step": 0.1},
    ] = 2.0,
) -> ScaleResult:
    """Scale an array, returning the result alongside its sum.

    Exercises multi-output ops: the NamedTuple field names become the names
    of the task's outputs.
    """
    scaled = image * factor
    return ScaleResult(scaled=scaled, total=float(scaled.sum()))


@op(env="minimal")
def scale_into(
    image: np.ndarray,
    result: Out[np.ndarray],
    factor: float = 2.0,
) -> None:
    """Computer form: fill a buffer the caller allocated.

    The caller owns ``result``, so nothing is sent back over the wire.
    """
    np.multiply(image, factor, out=result, casting="unsafe")


@op(env="minimal")
def slow_sum(image: np.ndarray, steps: int = 5) -> float:
    """Sum an array slowly, reporting progress and honoring cancellation."""
    total = 0.0
    chunks = np.array_split(image.ravel(), steps)
    for i, chunk in enumerate(chunks):
        if cancel_requested():
            break
        progress(f"Summing chunk {i + 1} of {steps}", i, steps)
        total += float(chunk.sum())
        time.sleep(0.05)
    return total


@op(env="minimal")
def probe() -> str:
    """Report which interpreter is actually executing this op."""
    import sys

    return f"{sys.executable} | py={sys.version.split()[0]} | numpy={np.__version__}"


class Findings(NamedTuple):
    labels: np.ndarray
    points: np.ndarray


@op(env="minimal")
def find_nothing(image: np.ndarray) -> Findings:
    """Return an empty result, as a segmentation that finds nothing would."""
    return Findings(
        labels=np.zeros_like(image, dtype=np.uint16),
        points=np.zeros((0, 3), dtype=np.int32),
    )
