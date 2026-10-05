# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["scikit-ops", "zarr>=3", "scikit-image"]
#
# [tool.uv.sources]
# scikit-ops = { path = "../..", editable = true }
# ///
"""Run a skop op on a zarr with too little memory, and watch it fail

Uses the membrane channel of IDR image idr0079A/9836998, a light-sheet
volume: z,y,x = 142 x 788 x 1584 uint8, 177 MB. The first run downloads
that channel into test_images/, in the same layout as the full copy
zarr-idr0079-download.py makes; later runs reuse it. These ops want numpy, so
before the op runs the host reads the zarr into memory and copies it into
shared memory for the worker (~2x the input). The op then makes a float32
copy and a float32 result (~8x more):

    gaussian   ~10x the input, ~1.8 GB
    frangi     a 3D Hessian: 6 float32 volumes on top, ~20x the input

Any machine has room for that, so the script caps its own memory. It
starts itself again under systemd-run, in a scope whose limit covers this
script and the worker it starts, and nothing else on the machine. A cap
between ~2x and ~10x the input fails inside the op; --cap none runs
without one, to see what the op really needs. The peak is printed either
way.

--skimage calls scikit-image on the zarr directly instead, in this
process, no skop and no worker. It does not pass the zarr through: it
converts to numpy too, and to float64, twice skop's float32. Run in this
process, a failure kills the script itself, so the shell prints "Killed"
rather than FAIL.

uv run docs/spec/zarr-idr0079-memory.py                 # gaussian, 1G cap
uv run docs/spec/zarr-idr0079-memory.py frangi
uv run docs/spec/zarr-idr0079-memory.py gaussian --cap none
uv run docs/spec/zarr-idr0079-memory.py --skimage --cap none
"""

import argparse
import http.client
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import zarr

import skop
from skop.ops.edges import frangi
from skop.ops.smooth import gaussian

OPS = {
    "gaussian": (gaussian, {"sigma": 2.0}),
    # one scale, so a failure is the Hessian and not the sweep
    "frangi": (frangi, {"sigma_min": 2.0, "sigma_max": 2.0}),
}
URL = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.3/idr0079A/9836998.zarr"
LOCAL = Path(__file__).resolve().parents[2] / "test_images" / "idr0079A-9836998.zarr"

parser = argparse.ArgumentParser()
parser.add_argument("op", nargs="?", default="gaussian", choices=OPS)
parser.add_argument("--cap", default="1G", help="memory cap, e.g. 1G, or none")
parser.add_argument("--skimage", action="store_true", help="call skimage, not skop")
args = parser.parse_args()


def fetch(key, tries=5):
    """Copy key from the IDR into LOCAL, unless it is already there.

    The IDR drops connections now and then, so a cut-off read is retried.
    """
    path = LOCAL / key
    if path.exists():
        return
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{URL}/{key}", timeout=60) as r:
                data = r.read()
            break
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


if "SKOP_MEMORY_SCOPE" not in os.environ:
    # Download first, so it isn't counted against the cap. Chunks are one
    # plane of one channel, keyed level/c/z/y/x; channel 0 is the membrane.
    for key in (".zgroup", ".zattrs", "0/.zarray"):
        fetch(key)
    for z in range(142):
        fetch(f"0/0/{z}/0/0")

    # Start again inside a scope; with no cap it still gives a peak to read.
    os.environ["SKOP_MEMORY_SCOPE"] = "1"
    cap = "infinity" if args.cap == "none" else args.cap
    os.execvp(
        "systemd-run",
        ["systemd-run", "--user", "--scope", "--quiet",
         "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0",
         sys.executable, *sys.argv],
    )  # fmt: skip


def scope():
    """The cgroup v2 directory this runs in."""
    path = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    return Path("/sys/fs/cgroup") / path.lstrip("/")


# a kill takes whatever is still buffered with it, so print a line at a time
sys.stdout.reconfigure(line_buffering=True)
group = scope()
cap = (group / "memory.max").read_text().strip()
fn, params = OPS[args.op]

with tempfile.TemporaryDirectory() as tmp:
    # Slicing a zarr array gives numpy, so copy the channel into a zarr of
    # its own: the op should be handed a zarr array, as a viewer would.
    image = zarr.open_group(LOCAL, mode="r")["0"]
    z = zarr.create_array(tmp, data=image[0], chunks=(1, *image.shape[-2:]))
    print(
        f"{args.op} on membrane channel: {z.shape} {z.dtype}, {z.nbytes / 1e6:.0f} MB"
    )
    print("memory cap:", "none" if cap == "max" else f"{int(cap) / 1e9:.1f} GB")

    print("via:", "skimage, in this process" if args.skimage else "skop worker")

    started = time.perf_counter()
    try:
        if args.skimage:
            from skimage import filters

            if args.op == "gaussian":
                out = filters.gaussian(z, sigma=2.0)
            else:
                out = filters.frangi(z, sigmas=[2.0])
        else:
            # closed on the way out: a live worker otherwise keeps Python
            # from exiting
            with skop.Runner() as runner:
                out = runner.run(fn, image=z, **params)
        print(f"OK    {type(out).__name__} {out.shape} {out.dtype}")
    except Exception as e:  # noqa: BLE001 -- a probe, the failure is the result
        print(f"FAIL  {type(e).__name__}: {e}")
    peak = int((group / "memory.peak").read_text()) / 1e9
    print(f"took {time.perf_counter() - started:.0f} s, peak {peak:.2f} GB")
