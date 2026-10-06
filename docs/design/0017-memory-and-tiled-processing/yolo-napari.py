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
"""Open a big image in napari at several scales, with YOLO ready to run

Each level is its own layer, scaled so they all line up: the same place in
the image at 2x, 4x, 8x fewer pixels. Pick a layer as the op's image and run
FastSAM on each to see which objects it finds at which scale -- a building
is an object at one level and a texture at another. The finest level shown
is selected; the others start hidden.

The layers are the zarr arrays themselves, so napari reads only what it
draws -- but a layer that is not multiscale draws all of itself when zoomed
out. The default is one level of about 50 Mpixels: em level 4 (5824 x
9024) or aerial level 2 (7000 x 7000 RGB). The finest levels can be shown
with --levels, but an op run on one gets all of it: FastSAM on em level 2
(840 Mpixels) took ~15 GB before resizing to 1024, and the kernel killed
it.

Needs yolo-download.py first, and the skop-napari checkout beside this one.

uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py em --levels 3 5 7
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial --op fastsam
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial --op jdll_yolo
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py --cap 3G
uv run docs/design/0017-memory-and-tiled-processing/yolo-napari.py aerial \\
    --boxes test_images/yolo-aerial-L4-fastsam-tile0.npy
"""

import argparse
import os
import re
import sys
from pathlib import Path

import napari
import numpy as np
import zarr
from skop_napari._panel import OpsPanel
from yolo_data import open_levels, path

WEIGHTS = Path(__file__).resolve().parents[4] / "i2k-2026" / "models" / "yolo26n.pt"

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="em", choices=["em", "aerial"])
parser.add_argument("--levels", type=int, nargs="+", help="default em 4, aerial 2")
parser.add_argument(
    "--op",
    default="yolo",
    choices=["yolo", "fastsam", "jdll_yolo"],
    help="op to have selected in the panel",
)
parser.add_argument(
    "--object-size",
    type=float,
    default=100,
    help="yolo's object_size, an area in pixels squared",
)
parser.add_argument(
    "--weights", type=Path, default=WEIGHTS, help="jdll_yolo's model, filled in"
)
parser.add_argument(
    "--boxes", type=Path, nargs="*", default=[], help=".npy files from yolo-run.py"
)
parser.add_argument("--cap", default="none", help="memory cap, e.g. 3G, or none")
args = parser.parse_args()

if args.cap != "none" and "SKOP_MEMORY_SCOPE" not in os.environ:
    # Start again inside a scope whose limit covers napari and its workers,
    # as zarr-idr0079-napari.py does. The kernel kills the biggest process in
    # the scope when it runs out, which may be napari rather than the op.
    os.environ["SKOP_MEMORY_SCOPE"] = "1"
    os.execvp(
        "systemd-run",
        ["systemd-run", "--user", "--scope", "--quiet",
         "-p", f"MemoryMax={args.cap}", "-p", "MemorySwapMax=0",
         sys.executable, *sys.argv],
    )  # fmt: skip

levels = open_levels(args.name)
# ~50 Mpixels each: the finest levels are too big to hand an op whole
shown = sorted(args.levels or [{"em": 4, "aerial": 2}[args.name]])
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
# Labels are "detect: yolo"; match the op's name exactly, since "yolo" is
# also in "jdll_yolo".
picks = [c for c in panel._picker.choices if c.split(": ")[-1] == args.op]
if picks:
    panel._picker.value = picks[0]
    for widget in panel._inputs.widgets:
        if widget.name == "weights":
            widget.value = args.weights
        elif widget.name == "object_size":
            widget.value = args.object_size
else:
    print(f"no op matching {args.op!r} in the panel")
viewer.window.add_dock_widget(panel, name="scikit-ops", area="right")
viewer.layers.selection.active = layers[-1]  # the finest level
print(f"credit: {group_attrs.get('credit', '')}")
if "SKOP_MEMORY_SCOPE" in os.environ:
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    group = Path("/sys/fs/cgroup") / group.lstrip("/")
    used = int((group / "memory.current").read_text()) / 1e9
    cap = int((group / "memory.max").read_text()) / 1e9
    print(f"napari up: {used:.2f} GB in use, cap {cap:.1f} GB")
napari.run()
