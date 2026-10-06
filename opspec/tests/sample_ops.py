"""Small ops for the tests to read specs off. Their bodies never run."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Annotated, NamedTuple

import numpy as np
from sample_results import Boxes

from opspec.op import Axes, Choices, Out, ParamsFor, op
from opspec.types import ImageOf, LabelsOf, PointsOf


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

    Deliberately carries no roles, and two outputs.
    """
    scaled = image * factor
    return ScaleResult(scaled=scaled, total=float(scaled.sum()))


@op(env="minimal")
def scale_into(
    image: ImageOf[np.ndarray],
    result: Out[ImageOf[np.ndarray]],
    factor: float = 2.0,
) -> None:
    """Computer form: fill a buffer the caller allocated."""
    np.multiply(image, factor, out=result, casting="unsafe")


@op(env="minimal")
def quadrants(
    image: Annotated[ImageOf[np.ndarray], Axes("y", "x")],
) -> LabelsOf[np.ndarray]:
    """A strictly 2-D op: label a plane's four quadrants."""
    return np.ones(image.shape, dtype=np.uint16)


@op(env="skimage")
def otsu(
    image: Annotated[ImageOf[np.ndarray], Axes(variadic=True)],
) -> LabelsOf[np.ndarray]:
    """A variadic op: one threshold for whatever it is handed."""
    return (image > image.mean()).astype(np.uint8)


class Findings(NamedTuple):
    labels: LabelsOf[np.ndarray]
    points: PointsOf[np.ndarray]


@op(env="minimal")
def find_nothing(image: ImageOf[np.ndarray]) -> Findings:
    """Two outputs, each with its own role."""
    return Findings(np.zeros(image.shape, np.uint16), np.zeros((0, 2)))


class Shape(Enum):
    ball = "ball"
    box = "box"
    diamond = "diamond"


@op(env="skimage")
def dilate(image: ImageOf[np.ndarray], shape: Shape = Shape.ball) -> np.ndarray:
    """An enum parameter, whose default travels as its value."""
    return image


def blur_fast(image):
    return image


def blur_careful(image, sigma=1.0):
    return image


@op
def workflow(
    image: np.ndarray,
    step: Annotated[
        Callable, Choices(fast=blur_fast, careful=blur_careful)
    ] = blur_fast,
    step_args: Annotated[dict | None, ParamsFor("step", binds="image")] = None,
) -> np.ndarray:
    """A workflow: no environment, and a menu of ops for one of its steps."""
    return step(image, **(step_args or {}))


@op(env="pytorch")
def detect(image: ImageOf[np.ndarray]) -> Boxes:
    """Returns a NamedTuple defined in another module, which this one never
    imports the field types of."""
    return Boxes(np.zeros((0, 4)))
