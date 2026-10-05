# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "napari[pyqt6]",
#     "scikit-ops",
#     "skop-napari",
#     "zarr>=3",
# ]
#
# [tool.uv.sources]
# scikit-ops = { path = "../..", editable = true }
# skop-napari = { path = "../../../skop-napari", editable = true }
# ///
"""Open one zarr layer in napari, with the skop plugin, and a memory cap

One layer, the membrane channel of idr0079A/9836998: z,y,x = 142 x 788 x
1584 uint8, 177 MB, as a plain zarr array -- no pyramid, no channels, no
labels. The first run makes it, test_images/idr0079A-membrane.zarr, from
the copy zarr-idr0079-download.py or zarr-idr0079-memory.py leaves in
test_images/; run one of those first.

The layer's data is the zarr array itself, so napari reads only the planes
it shows, and an op is handed the zarr, as zarr-idr0079-memory.py hands it
one from a script.

--cap starts napari again under systemd-run, in a scope whose limit
covers napari and the skop workers it starts. The memory in use is printed
once the viewer is up; the cap needs room above that for the input to be
copied for the worker (~2x the input, ~350 MB), and then not enough for
the op (gaussian ~8x more, ~1.5 GB; frangi ~18x, ~3 GB), so that the op
is what fails and not the copy. Untested; tune the cap from the printed
number.

uv run docs/spec/zarr-idr0079-napari.py
uv run docs/spec/zarr-idr0079-napari.py --cap 3G
"""

import argparse
import os
import sys
from pathlib import Path

import napari
import zarr

IMAGES = Path(__file__).resolve().parents[2] / "test_images"
FULL = IMAGES / "idr0079A-9836998.zarr"
MEMBRANE = IMAGES / "idr0079A-membrane.zarr"

parser = argparse.ArgumentParser()
parser.add_argument("--cap", default="none", help="memory cap, e.g. 3G, or none")
args = parser.parse_args()

if "SKOP_MEMORY_SCOPE" not in os.environ:
    # Made before the cap, so the copy isn't counted against it.
    if not MEMBRANE.exists():
        if not (FULL / "0" / ".zarray").exists():
            sys.exit(f"No copy at {FULL}; run zarr-idr0079-download.py first")
        source = zarr.open_group(FULL, mode="r")
        window = source.attrs["omero"]["channels"][0]["window"]
        plane = source["0"].shape[-2:]
        # c,z,y,x at level 0; channel 0 is the membrane
        z = zarr.create_array(MEMBRANE, data=source["0"][0], chunks=(1, *plane))
        z.attrs["contrast_limits"] = [window["start"], window["end"]]
        print(f"made {MEMBRANE}")

    if args.cap != "none":
        # Start again inside a scope whose limit covers napari and its workers.
        os.environ["SKOP_MEMORY_SCOPE"] = "1"
        os.execvp(
            "systemd-run",
            ["systemd-run", "--user", "--scope", "--quiet",
             "-p", f"MemoryMax={args.cap}", "-p", "MemorySwapMax=0",
             sys.executable, *sys.argv],
        )  # fmt: skip

membrane = zarr.open_array(MEMBRANE, mode="r")
viewer = napari.Viewer()
# contrast_limits given up front, or napari reads the data to guess them.
viewer.add_image(
    membrane,
    name="membrane",
    contrast_limits=membrane.attrs["contrast_limits"],
)

if "SKOP_MEMORY_SCOPE" in os.environ:
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    group = Path("/sys/fs/cgroup") / group.lstrip("/")
    used = int((group / "memory.current").read_text()) / 1e9
    cap = int((group / "memory.max").read_text()) / 1e9
    print(f"napari up: {used:.2f} GB in use, cap {cap:.1f} GB")
napari.run()
