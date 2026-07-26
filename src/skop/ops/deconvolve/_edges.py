"""Edge handling: deciding where the image has data, and extending it.

Padding and bad-pixel masking look like two chores but are one idea. An edge
is a place where no data was acquired; so is a saturated pixel. Both are
handled the same way -- held out of the HTones array, which is what tells the
iteration how much signal each location should have received.

This is identical whichever backend runs the iteration, and happens on the CPU
either way, so both share it. The two originals ported here had drifted apart
in exactly these steps; sharing them is what keeps that from happening again.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._pad import get_next_smooth, pad, pad_to_largest, unpad


class Padded(NamedTuple):
    """Arrays at their working size, and what it takes to get back."""

    image: np.ndarray
    psf: np.ndarray
    htones: np.ndarray
    original_size: tuple[int, ...]
    mask: np.ndarray | None
    mask_values: np.ndarray | None

    def crop_and_restore(self, estimate: np.ndarray) -> np.ndarray:
        """Undo ``pad_and_mask``: crop back, and put masked pixels back.

        A method rather than a free function because the padding is what
        knows how to reverse itself -- there is no way to pair the wrong
        result with the wrong padding.
        """
        if estimate.shape != self.original_size:
            estimate = unpad(estimate, self.original_size)
        if self.mask is not None:
            estimate = estimate * self.mask + self.mask_values
        return estimate


def pad_and_mask(
    image: np.ndarray,
    psf: np.ndarray,
    noncirc: bool,
    mask: np.ndarray | None,
    dtype=np.float64,
) -> Padded:
    """Hold out masked pixels, and pad the arrays to their working size.

    Note:
        When ``noncirc`` is off and the image and psf already agree, there is
        nothing to pad and only the masking applies.
    """
    image = np.asarray(image, dtype=dtype)
    psf = np.asarray(psf, dtype=dtype)
    original_size = image.shape

    mask_values = None
    if mask is None:
        htones = np.ones_like(image)
    else:
        mask = np.asarray(mask, dtype=dtype)
        if mask.shape != original_size:
            raise ValueError(
                f"mask shape {mask.shape} does not match image shape {original_size}"
            )
        # Masked pixels are set aside and restored afterwards. Folding the mask
        # into HTones is what makes them behave like edges.
        mask_values = image * (1.0 - mask)
        image = image * mask
        htones = mask.copy()

    if noncirc:
        extended = get_next_smooth(
            [image.shape[i] + 2 * (psf.shape[i] // 2) for i in range(image.ndim)]
        )
        image, _ = pad(image, extended, "constant")
        htones, _ = pad(htones, extended, "constant")
        psf, _ = pad(psf, extended, "constant")
    elif image.shape != psf.shape:
        # NB: the numpy original padded the psf up to the image, which
        # truncates a psf wider than the image. Both backends now pad each to
        # the larger of the two, as the cupy version did.
        image, psf = pad_to_largest(image, psf, "constant")
        htones, _ = pad(htones, image.shape, "constant")

    return Padded(image, psf, htones, original_size, mask, mask_values)
