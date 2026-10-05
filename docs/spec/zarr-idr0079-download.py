# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Copy idr0079A/9836998 from the IDR to test_images/

The small partner of jrc_hela-2: a 2-channel light-sheet volume (membrane
and nuclei), c,z,y,x = 2 x 142 x 788 x 1584 uint8, ~355 MB in memory, with
a 3D instance label image beside it. It fits in memory, so a chunked run
can be checked against a whole-array one; hela then shows it scales.

Copies the chunk files as they are, all six levels and the labels, so the
local copy is the same OME-Zarr (v0.3, zarr v2) as the remote one.

uv run docs/spec/zarr-idr0079-download.py

Already copied chunks are skipped, so it can be rerun after a stop.
"""

import http.client
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from math import ceil
from pathlib import Path

URL = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.3/idr0079A/9836998.zarr"
OUT = Path(__file__).resolve().parents[2] / "test_images" / "idr0079A-9836998.zarr"


def fetch(key, tries=5):
    """Return the bytes at key, or None if the remote has no such object.

    The IDR drops connections now and then, so a cut-off read is retried.
    """
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{URL}/{key}", timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # a chunk never written is fill value
                return None
            if attempt == tries - 1:
                raise
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2**attempt)


def copy_key(key):
    path = OUT / key
    if path.exists():
        return
    data = fetch(key)
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def copy_array(prefix):
    meta = json.loads(fetch(f"{prefix}/.zarray"))
    sep = meta.get("dimension_separator", ".")
    grid = [range(ceil(s / c)) for s, c in zip(meta["shape"], meta["chunks"])]
    keys = [f"{prefix}/" + sep.join(map(str, idx)) for idx in product(*grid)]
    copy_key(f"{prefix}/.zarray")
    print(f"{prefix}: {meta['shape']} {meta['dtype']}, {len(keys)} chunks")
    # 8, not 32 as for hela: the IDR drops connections under more
    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(copy_key, keys), 1):
            if i % 500 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)


def copy_image(prefix):
    """An OME multiscales group: its metadata, then every level."""
    for key in (".zgroup", ".zattrs"):
        copy_key(f"{prefix}{key}")
    attrs = json.loads((OUT / f"{prefix}.zattrs").read_text())
    for ms in attrs["multiscales"]:
        for d in ms["datasets"]:
            copy_array(f"{prefix}{d['path']}")


copy_image("")
copy_key("labels/.zgroup")
copy_key("labels/.zattrs")
for label in json.loads((OUT / "labels" / ".zattrs").read_text())["labels"]:
    copy_image(f"labels/{label}/")
print("done:", OUT)
