"""Training ops: input/truth pairs in, a model on disk out.

Training ops differ from the rest of skop in two ways, both settled in
design 0011. They take **paths, not arrays** -- a manifest of image files and
the label files that pair with them, one list each, matched by position. And
they **return a path**: a trained TensorFlow or torch model cannot cross the
worker boundary, so the op writes it and hands back where.

The caller resolves its own directory layout into those two lists. Nothing in
here knows what a patch directory looks like, and nothing in here should.
"""

from __future__ import annotations

from .stardist2d import train_stardist2d

__all__ = ["train_stardist2d"]
