"""Circle phantoms for the Cellpose training notebook.

One calibration image with a known circle per radius, and random images to
train on. Circles never overlap, so every one is its own label.
"""

import numpy as np


def _disc(r):
    """A filled circle of radius r, as a boolean array."""
    yy, xx = np.ogrid[-r:r, -r:r]
    return (yy ** 2 + xx ** 2) <= r * r


def _place(image, labels, cx, cy, r, index):
    """Put one circle in, unless it would touch a circle already there."""
    where = (slice(cy - r, cy + r), slice(cx - r, cx + r))
    disc = _disc(r)
    if labels[where][disc].any():
        return False
    labels[where][disc] = index
    image[where][disc] = 1.0
    return True


def calibration(radii, height=256, gap=24, noise=0.05, seed=0):
    """One circle per radius, in a row, smallest first."""
    width = gap + sum(2 * r + gap for r in radii)
    image = np.zeros((height, width), np.float32)
    labels = np.zeros((height, width), np.uint16)

    x = gap
    for index, r in enumerate(radii, start=1):
        _place(image, labels, x + r, height // 2, r, index)
        x += 2 * r + gap

    rng = np.random.default_rng(seed)
    return image + rng.normal(0, noise, image.shape).astype(np.float32), labels


def random(count, size=256, min_r=1, max_r=85, tries=80, noise=0.05, seed=0):
    """Images of non-overlapping circles at random radii."""
    rng = np.random.default_rng(seed)
    images, truths = [], []

    for _ in range(count):
        image = np.zeros((size, size), np.float32)
        labels = np.zeros((size, size), np.uint16)
        index = 1
        for _ in range(tries):
            r = int(rng.integers(min_r, max_r + 1))
            cx = int(rng.integers(r, size - r))
            cy = int(rng.integers(r, size - r))
            if _place(image, labels, cx, cy, r, index):
                index += 1
        images.append(image + rng.normal(0, noise, image.shape).astype(np.float32))
        truths.append(labels)

    return images, truths
