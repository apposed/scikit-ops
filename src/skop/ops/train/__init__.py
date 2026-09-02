"""Training ops: input/truth pairs in, a model on disk out.

Also what you want to know *before* training -- the receptive field of the
architecture you are about to build. It sits here rather than in a category of
its own because it takes the same architecture parameters the training op
does, and a change to one is a change to both.

Training ops differ from the rest of skop in two ways, both settled in
design 0011. They take **paths, not arrays** -- a manifest of image files and
the label files that pair with them, one list each, matched by position. And
they **return a path**: a trained TensorFlow or torch model cannot cross the
worker boundary, so the op writes it and hands back where.

The caller resolves its own directory layout into those two lists. Nothing in
here knows what a patch directory looks like, and nothing in here should.
"""

from __future__ import annotations

import numpy as np

from .stardist2d import receptive_field_stardist2d, train_stardist2d

__all__ = [
    "train_stardist2d",
    "receptive_field_stardist2d",
    "extent",
    "theoretical_receptive_field",
]


def extent(response) -> tuple[int, int]:
    """The height and width, in pixels, that an impulse response covers.

    Takes what a ``receptive_field_*`` op returns. Pure numpy, so it runs
    wherever the caller is rather than in the op's environment.
    """
    reached = np.asarray(response) > 0
    if not reached.any():
        return (0, 0)
    rows, cols = np.where(reached)
    return (int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1))


def theoretical_receptive_field(
    unet_n_depth: int = 3,
    grid_size_xy: int = 1,
    unet_kernel_size: int = 3,
    unet_pool: int = 2,
    unet_n_conv_per_depth: int = 2,
) -> int:
    """What the architecture predicts, for comparison with the measurement.

    Counted the usual way -- each convolution adds ``kernel - 1`` times the
    current stride, each pooling multiplies the stride -- down through the
    encoder, across the bottleneck and back up. ``grid`` multiplies the result
    because StarDist pools the input to grid size before the U-Net, so the
    whole network sees a coarser image.

    An estimate, and the measurement is the authority: it counts the final
    convolutions after the U-Net, which this does not, so it reads low.
    Compare the two, and when they disagree believe the impulse.
    """
    k, pool = unet_kernel_size, unet_pool
    r, stride = 1, 1
    for _ in range(unet_n_depth):
        for _ in range(unet_n_conv_per_depth):
            r += (k - 1) * stride
        r += (pool - 1) * stride
        stride *= pool
    for _ in range(unet_n_conv_per_depth):
        r += (k - 1) * stride
    for _ in range(unet_n_depth):
        stride //= pool
        for _ in range(unet_n_conv_per_depth):
            r += (k - 1) * stride
    return r * grid_size_xy
