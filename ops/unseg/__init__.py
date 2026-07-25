"""UNSEG: unsupervised segmentation of nuclei and cells.

Segments tissue images carrying both a nucleus marker and a cell membrane
marker, without training data.

Ported from https://github.com/ctrueden/unseg-fiji. The algorithm lives in
_algorithm.py, which is imported inside the op body because its own imports
(OpenCV, scikit-learn, scikit-image) are far too heavy for discovery.

  Uttam Lab. UNSEG: unsupervised segmentation of cells and their nuclei in
  complex tissue samples.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, NamedTuple

import numpy as np

from opkit import op, progress


class DistanceTransform(Enum):
    """How to compute distances when splitting touching objects."""

    geodesic = "GDT"
    euclidean = "DT"


class TernaryMethod(Enum):
    """How to resolve the three-way foreground/background/boundary decision."""

    argmax = "Argmax"
    posterior = "Posterior"


class Segmentation(NamedTuple):
    nuclei: np.ndarray
    cells: np.ndarray
    n_nuclei: int
    n_cells: int


@op(env="unseg-cv")
def unseg(
    image: np.ndarray,
    channel_axis: int = 0,
    nuclei_channel: int = 2,
    membrane_channel: int = 0,
    area_threshold: Annotated[int, {"min": 1, "max": 10000}] = 20,
    convexity_threshold: Annotated[int, {"min": 0, "max": 100}] = 4,
    cell_marker_threshold: Annotated[int, {"min": 0, "max": 255}] = 25,
    dist_tr: DistanceTransform = DistanceTransform.geodesic,
    sigma0: Annotated[int, {"min": 0, "max": 20}] = 3,
    k0: Annotated[int, {"min": 0, "max": 20}] = 1,
    r0: Annotated[int, {"min": 1, "max": 50}] = 5,
    pct: tuple[float, float] = (0.01, 0.01),
    nk: tuple[int, ...] = (5, 10, 20, 40),
    t0: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}] = 0.5,
    ternary_met: TernaryMethod = TernaryMethod.argmax,
    area_ratio_threshold: Annotated[
        float, {"min": 0.0, "max": 1.0, "step": 0.05}
    ] = 0.65,
    dilation_radius: Annotated[int, {"min": 1, "max": 50}] = 5,
) -> Segmentation:
    """Segment nuclei and cells in a two-marker tissue image.

    Args:
        image: Multichannel 2D image carrying both markers.
        channel_axis: Which axis of ``image`` holds the channels. Fiji and
            napari usually hand over (C, Y, X), hence the default of 0.
        nuclei_channel: Index of the nucleus marker channel, e.g. DAPI.
        membrane_channel: Index of the cell membrane marker channel, e.g.
            Na+/K+ ATPase.
        area_threshold: Smallest object area to keep, in pixels.
        convexity_threshold: How non-convex an object may be before UNSEG
            tries to split it.
        cell_marker_threshold: Membrane marker intensity below which a pixel
            is not considered part of a cell.
        dist_tr: Distance transform used when splitting touching objects.
        sigma0: Smoothing width for the empirical probability estimate.
        k0: Gradient-adaptive smoothing strength.
        r0: Radius of the local binary mask neighborhood.
        pct: Background removal percentiles, one per marker.
        nk: Cluster counts to try when building the global binary mask.
        t0: Threshold on the global binary mask.
        ternary_met: How to resolve the three-way mask.
        area_ratio_threshold: Nucleus-to-cell area ratio above which a cell is
            rejected.
        dilation_radius: How far to grow nuclei when seeding cells.

    Returns:
        Nucleus and cell label images, and the count of each.
    """
    from ._algorithm import nuclei_cell_segmentation

    nuclei_marker = np.take(image, nuclei_channel, axis=channel_axis)
    membrane_marker = np.take(image, membrane_channel, axis=channel_axis)
    if nuclei_marker.ndim != 2:
        raise ValueError(
            f"UNSEG segments 2D images, but selecting a channel of a "
            f"{image.ndim}D image left {nuclei_marker.ndim} dimensions. "
            f"Check channel_axis (currently {channel_axis})."
        )

    intensity = np.zeros((*nuclei_marker.shape, 2), dtype="float64")
    intensity[:, :, 0] = nuclei_marker
    intensity[:, :, 1] = membrane_marker

    progress(f"Segmenting {nuclei_marker.shape} image")
    nuc_seg, cel_seg, n_nuclei, n_cells = nuclei_cell_segmentation(
        intensity,
        area_threshold=area_threshold,
        convexity_threshold=convexity_threshold,
        cell_marker_threshold=cell_marker_threshold,
        dist_tr=dist_tr.value,
        sigma0=sigma0,
        k0=k0,
        r0=r0,
        pct=list(pct),
        nk=list(nk),
        t0=t0,
        ternary_met=ternary_met.value,
        # NB: no display in a worker process.
        visualization=False,
        area_ratio_threshold=area_ratio_threshold,
        dilation_radius=dilation_radius,
    )

    return Segmentation(
        nuclei=np.asarray(nuc_seg).astype(np.int32),
        cells=np.asarray(cel_seg).astype(np.int32),
        n_nuclei=int(n_nuclei),
        n_cells=int(n_cells),
    )
