# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["scikit-ops", "zarr>=3", "numpy"]
#
# [tool.uv.sources]
# scikit-ops = { path = "../../..", editable = true }
# ///
"""Run a box detector on one level of a big image, whole or in tiles

The scale question from yolo-sam-tiling.md, from a script. A detector sees
the image resized so its longest side is imgsz (1024 for FastSAM), so:

- the whole of a coarse level: big things are near the trained size, small
  things vanish
- tiles of a fine level, each about imgsz: small things are kept, big things
  are cut by tile edges

Try both and compare the box counts and the median box size it prints.

Tiles overlap by --overlap, the last one in a row or column is pinned to the
edge (JDLL's rule), and the boxes from all tiles are shifted back and merged
by NMS at IoU 0.5. The ops return no scores, so NMS keeps the bigger of two
overlapping boxes -- a stand-in; an object cut by a tile edge loses to its
whole self, which is roughly what is wanted.

The boxes are saved to test_images/ in that level's pixels; yolo-napari.py
--boxes shows them over the image.

Needs yolo-download.py first. The op runs in skop's pytorch environment,
built on first use.

uv run docs/design/0017-memory-and-tiled-processing/yolo-run.py aerial --level 4
uv run docs/design/0017-memory-and-tiled-processing/yolo-run.py aerial --level 2 --tile 1024
uv run docs/design/0017-memory-and-tiled-processing/yolo-run.py em --level 5 --op jdll_yolo
"""

import argparse
import time
from pathlib import Path

import numpy as np
from yolo_data import IMAGES, open_levels

import skop
from skop.ops.detect import fastsam, jdll_yolo

WEIGHTS = Path(__file__).resolve().parents[4] / "i2k-2026" / "models" / "yolo26n.pt"

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="em", choices=["em", "aerial"])
parser.add_argument("--level", type=int, help="pyramid level; default em 5, aerial 4")
parser.add_argument("--op", choices=["fastsam", "jdll_yolo"], default="fastsam")
parser.add_argument(
    "--tile", type=int, default=0, help="tile side in pixels; 0 = whole level"
)
parser.add_argument("--overlap", type=float, default=0.15)
parser.add_argument("--imgsz", type=int, help="op default if not given")
parser.add_argument("--conf", type=float, help="op default if not given")
parser.add_argument("--weights", type=Path, default=WEIGHTS, help="jdll_yolo only")
args = parser.parse_args()

levels = open_levels(args.name)
level = args.level if args.level is not None else {"em": 5, "aerial": 4}[args.name]
image = np.asarray(levels[level])
print(f"{args.name} level {level}: {image.shape} {image.dtype}")

fn = {"fastsam": fastsam, "jdll_yolo": jdll_yolo}[args.op]
params = {
    k: v for k, v in (("imgsz", args.imgsz), ("conf", args.conf)) if v is not None
}
if args.op == "jdll_yolo":
    params["weights"] = args.weights


def starts(size, tile, step):
    """Tile origins along one axis; the last is pinned to the edge."""
    if size <= tile:
        return [0]
    out = list(range(0, size - tile, step))
    return out + [size - tile]


def nms(boxes, iou=0.5):
    """Greedy NMS, bigger box first, since the ops return no scores."""
    if len(boxes) == 0:
        return boxes
    y0, x0, y1, x1 = boxes.T
    area = (y1 - y0) * (x1 - x0)
    order = np.argsort(-area)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        h = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        w = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        inter = h * w
        order = rest[inter / (area[i] + area[rest] - inter) <= iou]
    return boxes[keep]


h, w = image.shape[:2]
tile = args.tile or max(h, w)
step = max(1, int(tile * (1 - args.overlap)))
origins = [(y, x) for y in starts(h, tile, step) for x in starts(w, tile, step)]
print(
    f"{len(origins)} tile(s) of {min(tile, h)} x {min(tile, w)}, op {args.op} {params}"
)

found = []
start = time.perf_counter()
with skop.Runner() as runner:
    for i, (y, x) in enumerate(origins, 1):
        result = runner.run(fn, image=image[y : y + tile, x : x + tile], **params)
        b = np.asarray(result.boxes, dtype=float).reshape(-1, 4)
        found.append(b + [y, x, y, x])
        print(f"  tile {i}/{len(origins)} at {y},{x}: {len(b)} boxes", flush=True)
seconds = time.perf_counter() - start

boxes = np.concatenate(found)
merged = nms(boxes) if len(origins) > 1 else boxes
print(f"{len(boxes)} boxes, {len(merged)} after merging, {seconds:.1f} s")
if len(merged):
    side = np.sqrt((merged[:, 2] - merged[:, 0]) * (merged[:, 3] - merged[:, 1]))
    q = np.percentile(side, [10, 50, 90]).round()
    print(
        f"box side, level {level} pixels: 10% {q[0]:.0f}, median {q[1]:.0f}, 90% {q[2]:.0f}"
    )
    print(f"  = {q[1] * 2**level:.0f} px at level 0")

out = IMAGES / f"yolo-{args.name}-L{level}-{args.op}-tile{args.tile}.npy"
np.save(out, merged)
print("saved", out)
