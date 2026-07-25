"""3D nuclei segmentation with pretrained StarDist models.

Ported from https://github.com/ctrueden/starfun3d. Shares the 'stardist-tf'
environment with stardist2d.py.

The models come from Galindo et al. (2023), "3D Nuclei Segmentation By
Combining GAN Based Image Synthesis and Existing 3D Manual Annotations",
https://doi.org/10.1101/2023.12.06.570366, published at
https://zenodo.org/records/10518151. They total 36 MB, so they are fetched
into skop's asset cache on first use rather than carried in the repository.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, NamedTuple

import numpy as np

from skop import op, progress

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

    from skop import assets

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
