"""StarDist 2D and 3D nuclei segmentation on cells3d, viewed in napari.

Takes skimage's cells3d sample and segments its nuclei channel two ways:
stardist2d on the middle z slice, and starfun3d on the whole volume. Both
results are placed into empty arrays shaped like cells3d itself, so napari
scrolls them in step with the image. The 2D labels are blank everywhere except
that one slice -- scroll through z to watch them appear and disappear against
the 3D labels, which are present throughout.

Run it with napari available:

    uv run --with napari --with pyqt5 python interactive_tests/interactive_test_cells3d.py

The first run builds the 'stardist-tf' environment and downloads the
pretrained model weights.
"""

import numpy as np
from skimage.data import cells3d

import skop
from skop.ops.segment.stardist2d import PretrainedModel, stardist2d
from skop.ops.segment.starfun3d import Model, segment_nuclei

# cells3d is (z, c, y, x) with channel 0 membrane and channel 1 nuclei.
MEMBRANE_CHANNEL = 0
NUCLEI_CHANNEL = 1

print("Loading the cells3d sample...")
volume = cells3d()
print(f"  cells3d: {volume.shape} {volume.dtype}")

nuclei = volume[:, NUCLEI_CHANNEL]
middle = nuclei.shape[0] // 2
print(f"  nuclei channel: {nuclei.shape}, middle slice z={middle}")

runner = skop.Runner()

print("\nSending the middle nuclei slice to stardist2d...")
slice_labels = runner.run(
    stardist2d,
    image=nuclei[middle],
    model=PretrainedModel.fluo,
    prob_thresh=0.5,
    nms_thresh=0.4,
    on_progress=lambda event: print(f"    {event.message}", end="\r"),
)
print()

slice_labels = np.asarray(slice_labels)
print(f"  labels: {slice_labels.shape} {slice_labels.dtype}")
print(f"  objects found: {int(slice_labels.max())}")

print("\nSending the whole nuclei channel to starfun3d...")

nuclei_3d = runner.run(
    segment_nuclei,
    image=nuclei.astype(np.float32),
    model=Model.confocal,
    on_progress=lambda event: print(f"    {event.message}", end="\r"),
)


volume_labels = np.asarray(nuclei_3d.labels)
print(f"  labels: {volume_labels.shape} {volume_labels.dtype}")
print(f"  objects found: {int(volume_labels.max())}")

runner.close()

# Empty volumes shaped like the whole cells3d stack, so napari scrolls them in
# step with the image. The 2D result fills a single z slice of the nuclei
# channel; the 3D result fills that channel at every z.
labels_2d = np.zeros(volume.shape, dtype=np.uint16)
labels_2d[middle, NUCLEI_CHANNEL] = slice_labels
print(
    f"\n  2d labels volume: {labels_2d.shape} {labels_2d.dtype}, "
    f"{int((labels_2d > 0).sum())} labelled voxels in 1 of {labels_2d.shape[0]} slices"
)

labels_3d = np.zeros(volume.shape, dtype=np.uint16)
labels_3d[:, NUCLEI_CHANNEL] = volume_labels
print(
    f"  3d labels volume: {labels_3d.shape} {labels_3d.dtype}, "
    f"{int((labels_3d > 0).sum())} labelled voxels across all "
    f"{labels_3d.shape[0]} slices"
)

print("\nOpening napari...")
import napari

viewer = napari.Viewer()
viewer.add_image(volume, name="cells3d")
viewer.add_labels(labels_3d, name=f"starfun3d (c={NUCLEI_CHANNEL})")
viewer.add_labels(labels_2d, name=f"stardist2d (z={middle}, c={NUCLEI_CHANNEL})")
viewer.dims.set_point(0, middle)
viewer.dims.set_point(1, NUCLEI_CHANNEL)

napari.run()
