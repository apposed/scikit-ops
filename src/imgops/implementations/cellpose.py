import numpy as np

try:
    from cellpose import models
    dependencies_available = True
except ImportError:
    models = None
    dependencies_available = False

env_name = 'naplari-hacking'


def run_cellpose(
    image,
    diameter,
    flow_threshold,
    cellprob_threshold,
    use_gpu,
):
    img = np.asarray(image.ndarray()) if hasattr(image, "ndarray") else np.asarray(image)

    # collapse a small trailing channel axis to grayscale
    if img.ndim >= 3 and img.shape[-1] in (3, 4):
        gray = img[..., :3].mean(axis=-1)
    else:
        gray = img

    model = models.CellposeModel(gpu=use_gpu)
    result = model.eval(
        gray,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # cellpose returns (masks, flows, styles[, diams]) across versions.
    masks = result[0]
    return np.asarray(masks).astype(np.uint16)
