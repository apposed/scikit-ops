"""
Experiment: SkImage Otsu segmenter as a magicgui @guiclass "command".

Minimal stand-in for the dataclass-based OtsuSegmenter in
src/napari_ai_lab/Segmenters/GlobalSegmenters, but using magicgui's
``@guiclass`` so the parameter GUI is generated automatically.

Each command exposes:
* ``.gui``            -- a magicgui Container widget (auto-generated).
* ``.segment(image)`` -- run segmentation, return a uint16 label image.
* ``NAME``            -- label shown in the switcher combo.
"""

from __future__ import annotations

import numpy as np
from magicgui.experimental import guiclass

try:
    from skimage import filters, measure
    _is_skimage_available = True
except ImportError:
    filters = None
    measure = None
    _is_skimage_available = False

@guiclass
class OtsuCommand:
    """Otsu threshold -> connected-component labels."""

    NAME = "Otsu (skimage)"

    invert: bool = False
    label_objects: bool = True

    def are_dependencies_available(self) -> bool:
        return _is_skimage_available

    def environment_path(self) -> str | None:
        """Return the pinned remote env path for this command, or None."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "appose"))
        from execute_appose import get_registry
        env = get_registry().resolve(type(self).__name__)
        return env.path if env is not None else None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Threshold with Otsu, optionally label connected components."""
        gray = _to_gray(image)
        thresh = filters.threshold_otsu(gray)
        mask = gray > thresh
        if self.invert:
            mask = ~mask
        if self.label_objects:
            return measure.label(mask).astype(np.uint16)
        return mask.astype(np.uint16)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis to grayscale."""
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    return image
