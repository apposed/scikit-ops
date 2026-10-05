# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "napari[pyqt6]",
#     "scikit-ops",
#     "skop-napari",
#     "zarr>=3",
#     "numpy",
# ]
#
# [tool.uv.sources]
# scikit-ops = { path = "../../..", editable = true }
# skop-napari = { path = "../../../../skop-napari", editable = true }
# ///
"""Open a big image in napari at several scales, with FastSAM ready to run

Each level is its own layer, scaled so they all line up: the same place in
the image at 2x, 4x, 8x fewer pixels. Pick a layer as the op's image and run
FastSAM on each to see which objects it finds at which scale -- a building
is an object at one level and a texture at another. The finest level shown
is selected; the others start hidden.

The layers are the zarr arrays themselves, so napari reads only what it
draws -- but a layer that is not multiscale draws all of itself when zoomed
out. The default is the finest level alone, which for em is 23296 x 36096,
~1.7 GB, and for aerial 28k x 28k RGB, ~2.4 GB: slow to open, and running an
op on it hands the op all of it.

Needs yolo-download.py first, and the skop-napari checkout beside this one.

uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py em --levels 3 5 7
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial --op jdll_yolo
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial \\
    --boxes test_images/yolo-aerial-L4-fastsam-tile0.npy
"""

import argparse
import re
from pathlib import Path

import napari
import numpy as np
import zarr
from skop_napari._panel import OpsPanel
from yolo_data import open_levels, path

WEIGHTS = Path(__file__).resolve().parents[4] / "i2k-2026" / "models" / "yolo26n.pt"

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="em", choices=["em", "aerial"])
parser.add_argument(
    "--levels", type=int, nargs="+", help="default the finest level only"
)
parser.add_argument(
    "--op",
    default="fastsam",
    choices=["fastsam", "jdll_yolo"],
    help="op to have selected in the panel",
)
parser.add_argument(
    "--weights", type=Path, default=WEIGHTS, help="jdll_yolo's model, filled in"
)
parser.add_argument(
    "--boxes", type=Path, nargs="*", default=[], help=".npy files from yolo-run.py"
)
args = parser.parse_args()

levels = open_levels(args.name)
shown = sorted(args.levels or [min(levels)])  # the finest there is
missing = [k for k in shown if k not in levels]
if missing:
    raise SystemExit(f"no level {missing} in {args.name}; there are {list(levels)}")

viewer = napari.Viewer(title=f"{args.name} at levels {shown}")
group_attrs = dict(zarr.open_group(path(args.name), mode="r").attrs)

layers = []
for k in reversed(shown):  # coarsest at the bottom
    data = levels[k]
    kwargs = {"rgb": data.ndim == 3}
    if "contrast_limits" in group_attrs:
        # given up front, or napari reads the whole level to guess them
        kwargs["contrast_limits"] = group_attrs["contrast_limits"]
    elif data.ndim == 3:
        kwargs["contrast_limits"] = [0, 255]
    layers.append(
        viewer.add_image(
            data,
            name=f"level {k} ({data.shape[0]} x {data.shape[1]})",
            scale=(2**k, 2**k),
            visible=k == shown[0],
            **kwargs,
        )
    )

for f in args.boxes:
    match = re.search(r"-L(\d+)-", f.name)
    k = int(match.group(1)) if match else shown[0]
    b = np.load(f)
    rects = [np.array([[y0, x0], [y1, x1]]) for y0, x0, y1, x1 in b]
    viewer.add_shapes(
        rects,
        shape_type="rectangle",
        name=f.stem,
        scale=(2**k, 2**k),
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )

panel = OpsPanel(viewer)
picks = [c for c in panel._picker.choices if args.op in c.lower()]
if picks:
    panel._picker.value = picks[0]
    for widget in panel._inputs.widgets:
        if widget.name == "weights":
            widget.value = args.weights
else:
    print(f"no op matching {args.op!r} in the panel")
viewer.window.add_dock_widget(panel, name="scikit-ops", area="right")
viewer.layers.selection.active = layers[-1]  # the finest level
print(f"credit: {group_attrs.get('credit', '')}")
napari.run()
