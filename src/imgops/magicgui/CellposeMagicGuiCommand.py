"""
Experiment: CellposeSAM segmenter as a magicgui @guiclass "command".

Minimal stand-in for the dataclass-based CellposeSegmenter -- 2D only,
CellposeSAM ("cpsam") only, no training.  Uses magicgui's ``@guiclass``
so the parameter GUI is generated automatically.

Exposes the same tiny interface as the Otsu command:
* ``.gui``            -- auto-generated magicgui Container widget.
* ``.segment(image)`` -- run segmentation, return a uint16 label image.
* ``NAME``            -- label shown in the switcher combo.

Cellpose is imported lazily inside ``segment`` so this file (and the
switcher panel) still load in environments without cellpose installed.
"""

from __future__ import annotations

import numpy as np
from magicgui.experimental import guiclass

try:
    from cellpose import models
    _is_cellpose_available = True
except ImportError:
    models = None
    _is_cellpose_available = False

@guiclass
class CellposeCommand:
    """CellposeSAM 2D inference."""

    NAME = "CellposeSAM"

    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    use_gpu: bool = True 

    def are_dependencies_available(self) -> bool:
        return _is_cellpose_available

    def environment_path(self) -> str | None:
        """Return the pinned remote env path for this command, or None."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "appose"))
        from execute_appose import get_registry
        env = get_registry().resolve(type(self).__name__)
        return env.path if env is not None else None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run CellposeSAM and return a uint16 label image."""

        gray = _to_gray(image)

        model = models.CellposeModel(gpu=self.use_gpu)
        result = model.eval(
            gray,
            diameter=self.diameter if self.diameter > 0 else None,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        # cellpose returns (masks, flows, styles[, diams]) across versions.
        masks = result[0]
        return np.asarray(masks).astype(np.uint16)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis to grayscale."""
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    return image
