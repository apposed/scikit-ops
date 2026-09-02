import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(11)
size = 224
yy, xx = np.mgrid[0:size, 0:size]

centers = [
    (y + rng.integers(-5, 6), x + rng.integers(-5, 6))
    for y in range(28, size - 20, 48)
    for x in range(28, size - 20, 48)
]
nuclei = np.zeros((size, size))
for cy, cx in centers:
    # NB: heterogeneous sizes on purpose. UNSEG multi-Otsu-thresholds the
    # small-object areas, which needs them to actually differ.
    sigma = rng.uniform(5.0, 11.0)
    nuclei += rng.uniform(0.6, 1.0) * np.exp(
        -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    )

membrane = np.zeros((size, size))
for edge in range(0, size, 48):
    membrane[max(edge - 2, 0) : edge + 3, :] = 1.0
    membrane[:, max(edge - 2, 0) : edge + 3] = 1.0

image = np.zeros((3, size, size), dtype=np.uint8)
image[0] = (np.clip(membrane + rng.normal(0, 0.10, (size, size)), 0, 1) * 220).astype(
    np.uint8
)
nuclei = np.clip(nuclei + rng.normal(0, 0.06, (size, size)), 0, 1)
image[2] = (nuclei / nuclei.max() * 255).astype(np.uint8)

from csbdeep.utils import normalize as normalize_percentile
from stardist.models import StarDist2D

normalize = True


def to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis, leaving other axes alone.

    A trailing extent of 3 or 4 is taken to be RGB(A); anything else is
    assumed to be spatial and returned untouched.
    """
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1).squeeze()
    return image


# The H&E model wants the colour channels the fluorescence model discards.
x = to_gray(image)

x = x.astype(np.float32)

if normalize:
    x = normalize_percentile(x, 1, 99.8)

net = StarDist2D.from_pretrained("2D_versatile_fluo")

stardist_result = net.predict_instances(x)

plt.imshow(stardist_result, cmap="nipy_spectral")
plt.show()
