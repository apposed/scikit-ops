# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Copy levels 2-10 of idr0083A/9822152 from the IDR to test_images/

One big 2D EM image, t,c,z,y,x = 1 x 1 x 1 x 93184 x 144384 uint16 at full
resolution, 11 levels. Level 2 is 23296 x 36096, ~1.7 GB in memory: big
enough to need tiling for YOLO, small enough to download.

Levels 0 and 1 (27 GB and 6.7 GB raw) are left on the IDR, to be read from
the URL. The group metadata is copied as it is, so .zattrs still lists them;
a reader that opens every level of the local copy will not find 0 and 1.

uv run docs/design/0017-memory-and-tiled-processing/zarr-idr0083-download.py

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

URL = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr"
OUT = Path(__file__).resolve().parents[3] / "test_images" / "idr0083A-9822152.zarr"
LEVELS = range(2, 11)


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
    with ThreadPoolExecutor(8) as pool:  # the IDR drops connections under more
        for i, _ in enumerate(pool.map(copy_key, keys), 1):
            if i % 100 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)


for key in (".zgroup", ".zattrs"):
    copy_key(key)
for level in LEVELS:
    copy_array(str(level))
print("done:", OUT)
