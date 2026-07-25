import numpy as np
import appose

try:
    from csbdeep.utils import normalize as _normalize
    from stardist.models import StarDist2D
    dependencies_available = True
except ImportError:
    _normalize = None
    StarDist2D = None
    dependencies_available = False

env_name = 'stardist'

def run_stardist(
    image,
    model_name,
    prob_thresh,
    nms_thresh,
    normalize,
):
    img = np.asarray(image.ndarray()) if hasattr(image, "ndarray") else np.asarray(image)

    # 2D_versatile_he expects RGB; the fluo model expects grayscale.
    if model_name == "2D_versatile_fluo":
        if img.ndim >= 3 and img.shape[-1] in (3, 4):
            x = img[..., :3].mean(axis=-1)
        else:
            x = img
    else:
        x = img

    if normalize:
        x = _normalize(x, 1, 99.8)

    model = StarDist2D.from_pretrained(model_name)
    labels, _ = model.predict_instances(
        x,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    labels = np.asarray(labels).astype(np.uint16)

    return labels
