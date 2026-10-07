# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "napari[pyqt6]",
#     "scikit-ops",
#     "skop-napari",
#     "scipy",
# ]
#
# [tool.uv.sources]
# scikit-ops = { path = "../../..", editable = true }
# skop-napari = { path = "../../../../skop-napari", editable = true }
# ///
"""Tiled Richardson-Lucy on the GPU, in napari, on a simulated volume

Five spheres blurred by a Gibson-Lanni PSF, with no noise, so a seam
between tiles shows: one in the middle, where the tiles meet, and one in
each corner, each a different brightness. A radius is a fraction of every
axis -- 20% in the middle, 10% in the corners -- so they are ellipsoids
that fit the volume whatever its shape. 128 x 1024 x 1536 by default: 0.8 GB in float32, and about
4 tiles (2 x 2) on an 8 GB GPU.

Opens napari with the skop panel set up: richardson_lucy_cupy, the blurred
spheres as the image, the PSF as the psf, noncirc ticked. Press Run. The
panel shows the plan before you do; type a smaller budget (2G) or a tile
size to change it.

The PSF is 64 x 64 x 32 (y, x, z), made in the sdeconv environment. The
truth -- the spheres before blurring -- is a hidden layer, to compare with.

D=docs/design/0017-memory-and-tiled-processing
uv run $D/rlcu-tiled-napari-sim.py
uv run $D/rlcu-tiled-napari-sim.py --shape 128 1536 1536   # 8 tiles
"""

import argparse
import time

import napari
import numpy as np
import scipy.fft
from scipy.signal import fftconvolve
from skop_napari import OpsPanel

import skop
from skop.ops.kernels.gibson_lanni import gibson_lanni

parser = argparse.ArgumentParser()
parser.add_argument("--shape", type=int, nargs=3, default=(128, 1024, 1536))
args = parser.parse_args()

AXES = ("z", "y", "x")

# -- the PSF ------------------------------------------------------------------

# From a runner of its own, before the panel exists: the panel's runner
# reports to the GUI, which is not running yet, and would wait on it.
print("making a Gibson-Lanni PSF, 64 x 64 x 32, in the sdeconv environment")
with skop.Runner() as runner:
    psf = runner.run(gibson_lanni, xy_size=64, z_size=32)

# -- the spheres --------------------------------------------------------------

started = time.perf_counter()
truth = np.zeros(args.shape, np.float32)
# (y, x) of the centre, as fractions; radius as a fraction; brightness.
SPHERES = [
    ((0.5, 0.5), 0.2, 100.0),
    ((0.2, 0.2), 0.1, 50.0),
    ((0.2, 0.8), 0.1, 100.0),
    ((0.8, 0.2), 0.1, 150.0),
    ((0.8, 0.8), 0.1, 200.0),
]
z, y, x = (np.arange(n) / n for n in args.shape)
for (cy, cx), r, brightness in SPHERES:
    inside = (
        ((z[:, None, None] - 0.5) / r) ** 2
        + ((y[None, :, None] - cy) / r) ** 2
        + ((x[None, None, :] - cx) / r) ** 2
    ) <= 1
    truth[inside] = brightness
    del inside

with scipy.fft.set_workers(-1):
    image = fftconvolve(truth, psf, mode="same").astype(np.float32)
np.maximum(image, 0, out=image)  # the FFT leaves tiny negatives; RL wants none
print(
    f"{len(SPHERES)} spheres in {image.shape}, {image.nbytes / 1e9:.1f} GB "
    f"float32, blurred in {time.perf_counter() - started:.0f} s"
)

# -- napari -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(truth, name="truth", visible=False, axis_labels=AXES)
viewer.add_image(psf, name="psf (Gibson-Lanni)", visible=False, axis_labels=AXES)
# Named axes, or the panel leaves the op's optional z empty and runs it once
# per plane: a 3-D PSF on 2-D planes.
viewer.add_image(image, name="blurred", axis_labels=AXES)

panel = OpsPanel(viewer)
viewer.window.add_dock_widget(panel, name="Ops (scikit-ops)")
panel._picker.value = next(
    label
    for label, spec in panel._by_label.items()
    if spec.name.endswith(":richardson_lucy_cupy")
)
widgets = {widget.name: widget for widget in panel._inputs.widgets}


def choose(name: str, layer: str) -> None:
    """Pick a layer in a layer combo, by its name.

    Not by value: the choices are layer data, and magicgui finds a value by
    ==, which an array cannot answer.
    """
    combo = widgets[name].native
    combo.setCurrentIndex(combo.findText(f"{layer} (data)"))  # napari's label


choose("image", "blurred")
choose("psf", "psf (Gibson-Lanni)")
widgets["noncirc"].value = True
print(f"ready: richardson_lucy_cupy, noncirc on; {panel._tiles.value}")
napari.run()
