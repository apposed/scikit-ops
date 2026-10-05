# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "numpy", "tifffile", "imagecodecs"]
# ///
"""Get one of the two big images the yolo-*.py scripts use

    em      levels 2-10 of IDR idr0083A/9822152, through
            zarr-idr0083-download.py, ~2 GB
    aerial  a 28k x 28k drone mosaic of Dhaka from OpenAerialMap, 174 MB

Either way the result is test_images/yolo-<name>.zarr, one plain 2D array
per level; see yolo_data.py. Level 0 of aerial is read whole to write it,
~2.4 GB in memory, once.

uv run docs/design/0017-memory-and-tiled-processing/yolo-download.py em
uv run docs/design/0017-memory-and-tiled-processing/yolo-download.py aerial

Rerunning skips what is already there.
"""

import argparse
import runpy
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
import zarr
from yolo_data import DATA, IMAGES, path

HERE = Path(__file__).resolve().parent
CHUNK = 1024

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="em", choices=list(DATA))
args = parser.parse_args()
out = path(args.name)


def write(group, level, data):
    """One level as a plain 2D (or 2D RGB) array, chunked for napari."""
    chunks = (CHUNK, CHUNK, *data.shape[2:])
    group.create_array(str(level), data=data, chunks=chunks)
    print(f"  level {level}: {data.shape} {data.dtype}", flush=True)


def em():
    runpy.run_path(str(HERE / "zarr-idr0083-download.py"), run_name="__main__")
    source = zarr.open_group(IMAGES / "idr0083A-9822152.zarr", mode="r")
    levels = DATA["em"]["levels"]
    group = zarr.open_group(out, mode="w")
    for k in levels:
        # t,c,z,y,x -> y,x, and big-endian uint16 -> native
        write(group, k, np.asarray(source[str(k)][0, 0, 0]).astype(np.uint16))
    coarse = np.asarray(group[str(levels[-3])])
    lo, hi = np.percentile(coarse[coarse > 0], [0.5, 99.5])
    group.attrs.update(levels=levels, contrast_limits=[float(lo), float(hi)])


def aerial():
    tif = IMAGES / "yolo-aerial-hatirjheel.tif"
    if not tif.exists():
        part = tif.with_suffix(".part")
        print(f"downloading {DATA['aerial']['source']}")
        urllib.request.urlretrieve(DATA["aerial"]["source"], part)
        part.rename(tif)
    group = zarr.open_group(out, mode="w")
    with tifffile.TiffFile(tif) as t:
        levels = t.series[0].levels
        for k, level in enumerate(levels):
            write(group, k, level.asarray())
    group.attrs.update(levels=list(range(len(levels))))


IMAGES.mkdir(exist_ok=True)
if (out / "zarr.json").exists() and "levels" in zarr.open_group(out).attrs:
    print(f"already there: {out}")
else:
    print(f"making {out}")
    {"em": em, "aerial": aerial}[args.name]()
    zarr.open_group(out).attrs["credit"] = DATA[args.name]["credit"]
    print("done:", out)
