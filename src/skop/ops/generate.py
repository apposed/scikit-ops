"""Synthetic data, for exercising ops that have nothing to segment yet.

Kept out of ``skop.ops.segment`` because generating a volume is not
segmenting one, and because these ops run in a light environment rather than
the deep learning stacks the segmenters need.
"""

from __future__ import annotations

import numpy as np

from skop import op, progress
from skop.types import ImageData


@op(env="skimage")
def synthetic_nuclei(
    size_z: int = 64,
    size_y: int = 256,
    size_x: int = 256,
    n_nuclei: int = 8,
    radius_z_min: float = 2.5,
    radius_z_max: float = 4.5,
    radius_xy_min: float = 16.0,
    radius_xy_max: float = 22.0,
    noise_level: int = 6553,
    seed: int | None = None,
) -> ImageData:
    """Generate a synthetic 3D volume of anisotropic Gaussian blob nuclei.

    The default radii are centered on the confocal model's training data, so
    the result is something ``skop.ops.segment.segment_nuclei`` should be
    able to find.

    Args:
        size_z: Volume extent along Z.
        size_y: Volume extent along Y.
        size_x: Volume extent along X.
        n_nuclei: How many nuclei to place.
        radius_z_min: Smallest nucleus radius along Z, as a Gaussian sigma.
        radius_z_max: Largest nucleus radius along Z.
        radius_xy_min: Smallest nucleus radius in XY.
        radius_xy_max: Largest nucleus radius in XY.
        noise_level: Amplitude of the uniform background noise.
        seed: Seed for the placement of nuclei. None places them at random,
            which means the number that can actually be told apart varies.

    Returns:
        A uint16 volume in (Z, Y, X) order.
    """
    from scipy.ndimage import gaussian_filter

    shape = (size_z, size_y, size_x)
    rng = np.random.default_rng(seed)
    volume = (rng.random(shape) * noise_level).astype(np.uint16)

    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]

    for i in range(n_nuclei):
        progress(f"Placing nucleus {i + 1} of {n_nuclei}", i, n_nuclei)
        z = rng.integers(0, shape[0])
        y = rng.integers(0, shape[1])
        x = rng.integers(0, shape[2])
        radius_z = rng.uniform(radius_z_min, radius_z_max)
        radius_xy = rng.uniform(radius_xy_min, radius_xy_max)
        intensity = rng.uniform(32768, 65535)

        dist_sq = (
            (zz - z) ** 2 / (2 * radius_z**2)
            + (yy - y) ** 2 / (2 * radius_xy**2)
            + (xx - x) ** 2 / (2 * radius_xy**2)
        )
        blob = intensity * np.exp(-dist_sq)
        volume = np.clip(np.maximum(volume, blob), 0, 65535).astype(np.uint16)

    return gaussian_filter(volume.astype(np.float32), sigma=0.5).astype(np.uint16)
