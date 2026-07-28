"""What every box detector returns.

One class, shared by every detector op, so that "these ops are substitutable"
is a fact rather than a convention -- a workflow choosing between them gets
the same fields and the same output names either way.

A one-field NamedTuple rather than a bare return annotation, because the field
name becomes the output name (0001): a front end labels the layer "boxes"
rather than "result", and adding a second output later is additive.

No classes: these detectors are class-agnostic by design, and a column of
zeros pretending to be class IDs would be worse than its absence.

No scores either, for now, though every detector here computes them. A
confidence per box is not a layer -- it is a *feature* of one, and skop has no
way to say that yet. Returning it as a bare array made a front end guess, and
napari guessed "image", which a 1-D array cannot be. Rather than encode a
guess, the value is dropped until the question is answered. See
docs/spec/per-object-features.md.
"""

from __future__ import annotations

from typing import NamedTuple

from skop.types import BoxesData


class Boxes(NamedTuple):
    #: (N, 4) as [min_y, min_x, max_y, max_x], in image coordinates.
    boxes: BoxesData
