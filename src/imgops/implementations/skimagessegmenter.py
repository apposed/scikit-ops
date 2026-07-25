import numpy as np

try:
    from skimage import filters, measure
    dependencies_available = True
except ImportError:
    filters = None
    measure = None
    dependencies_available = False

env_name = 'naplari-hacking'


def run_otsu(
    image,
    invert,
    label_objects,
):
    img = np.asarray(image.ndarray()) if hasattr(image, "ndarray") else np.asarray(image)

    # collapse a small trailing channel axis to grayscale
    if img.ndim >= 3 and img.shape[-1] in (3, 4):
        gray = img[..., :3].mean(axis=-1)
    else:
        gray = img

    thresh = filters.threshold_otsu(gray)
    mask = gray > thresh
    if invert:
        mask = ~mask
    if label_objects:
        return measure.label(mask).astype(np.uint16)
    return mask.astype(np.uint16)
