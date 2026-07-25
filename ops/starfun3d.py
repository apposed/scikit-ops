"""3D nuclei segmentation with pretrained StarDist models.

Ported from https://github.com/ctrueden/starfun3d. Shares the 'stardist-tf'
environment with ops/stardist2d.py.

The models come from Galindo et al. (2023), "3D Nuclei Segmentation By
Combining GAN Based Image Synthesis and Existing 3D Manual Annotations",
https://doi.org/10.1101/2023.12.06.570366, published at
https://zenodo.org/records/10518151. They total 36 MB, so they are fetched
into opkit's asset cache on first use rather than carried in the repository.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, NamedTuple

import numpy as np

from opkit import op, progress

_ZENODO = "https://zenodo.org/records/10518151/files/model_{name}.zip?download=1"


class Model(Enum):
    """Pretrained StarDist3D models, one per acquisition modality.

    Average nucleus size in the training data, in pixels:
    confocal [x=39, y=39, z=7], sospim [x=27, y=28, z=10],
    spinning [x=39, y=39, z=7].
    """

    confocal = "confocal"
    sospim = "sospim"
    spinning = "spinning"


class Nuclei(NamedTuple):
    labels: np.ndarray
    points: np.ndarray


@op(env="stardist-tf")
def segment_nuclei(
    image: np.ndarray,
    model: Model = Model.confocal,
    prob_thresh: Annotated[
        float | None,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = None,
    nms_thresh: Annotated[
        float | None,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.05},
    ] = None,
    normalize: bool = True,
) -> Nuclei:
    """Segment nuclei in a 3D volume.

    Args:
        image: Volume to segment, in (Z, Y, X) order.
        model: Which pretrained model to use; pick the one matching how the
            data was acquired.
        prob_thresh: Object probability threshold. None uses the model's own.
        nms_thresh: Non-maximum suppression threshold. None uses the model's.
        normalize: Whether to percentile-normalize the input first.

    Returns:
        A label volume, and the detected nucleus centers as (N, 3) points.
    """
    from csbdeep.utils import normalize as normalize_percentile
    from stardist.models import StarDist3D

    from opkit import assets

    model_dir = assets.unzip_from_url(
        _ZENODO.format(name=model.value),
        f"starfun3d/{model.value}",
        marker="config.json",
    )

    progress(f"Loading model {model.value}")
    # NB: StarDist3D locates a model as basedir/name, so the model's own
    # directory name must be used -- the original starfun3d passed a constant
    # here, and so loaded the confocal model whichever model was asked for.
    net = StarDist3D(None, name=model_dir.name, basedir=str(model_dir.parent))

    x = normalize_percentile(image, 1, 99.8) if normalize else image

    predict_kwargs = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    progress("Predicting instances")
    labels, details = net.predict_instances(x, **predict_kwargs)

    points = np.asarray(details.get("points", np.zeros((0, 3))))
    return Nuclei(
        labels=np.asarray(labels).astype(np.uint16),
        points=points.astype(np.int32),
    )


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
) -> np.ndarray:
    """Generate a synthetic 3D volume of anisotropic Gaussian blob nuclei.

    The default radii are centered on the confocal model's training data, so
    the result is something ``segment_nuclei`` should be able to find.

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
