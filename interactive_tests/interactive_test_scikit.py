import matplotlib.pyplot as plt
import numpy as np

from opkit import runner
from ops import cellpose, stardist2d, unseg

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

print("Sending unseg op to the runner...")
result = runner.run(unseg.unseg, image=image)

print(f"result.n_nuclei: {result.n_nuclei}, expected: {len(centers)}")
print(f"result.n_cells: {result.n_cells}, expected: {len(centers)}")
print(f"result.nuclei.shape: {result.nuclei.shape}, expected: {(size, size)}")
print(f"result.cells.shape: {result.cells.shape}, expected: {(size, size)}")

print("\nSending cellpose op to the runner...")
cellpose_result = runner.run(cellpose.cellpose, image=image)
print(f"cellpose result: {cellpose_result}")

print("\nSending stardist2d op to the runner...")

stardist_result = runner.run(stardist2d.stardist2d, image=image[2])
# stardist_result = stardist2d.stardist2d(image=image)

print(f"stardist2d result: {stardist_result}")

# Create a 2x2 grid visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))


def show_rgb(ax, arr, title):
    """RGB needs uint8 in 0-255 or float in 0-1; rescale to float 0-1."""
    hi = arr.max()
    if hi > 0:
        arr = arr / hi
    ax.imshow(np.clip(arr, 0, 1))
    ax.set_title(title)
    ax.axis("off")


def show_labels(ax, arr, title):
    """Label images: pin vmin/vmax so a mostly-zero image still shows up."""
    ax.imshow(
        arr,
        cmap="nipy_spectral",
        vmin=0,
        vmax=max(int(arr.max()), 1),
        interpolation="nearest",
    )
    ax.set_title(f"{title} (max label {int(arr.max())})")
    ax.axis("off")


# Original image (channel-first, so transpose to get HxWx3)
show_rgb(axes[0, 0], np.transpose(image, (1, 2, 0)), "Original Image")

# UNSEG results
show_labels(axes[0, 1], result.nuclei, "UNSEG")

# Stardist2d results
show_labels(axes[1, 0], stardist_result, "StarDist2D")

# Cellpose results
show_labels(axes[1, 1], cellpose_result, "Cellpose")

plt.tight_layout()
plt.show()
