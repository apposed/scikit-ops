# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr", "icechunk", "scikit-image"]
# ///
"""  Test skimage functions on a Zarray 

Zarr Arrays do not have all numpy methods.

Functions that call np.asarray first work, but could explode memory and fail 

Functions that do not convert to numpy can fail when calling numpy methods (copy, reshape)

uv run docs/design/0017-memory-and-tiled-processing/zarr-skimage.py

The coins image is written to an icechunk repository in a temporary
directory, which is deleted at the end.
"""

import tempfile
import warnings

import icechunk
import zarr
from skimage import data, filters, measure, morphology, segmentation, transform

warnings.filterwarnings("ignore")


def run(f, *args):
    """Call f(*args) and print whether it worked."""
    try:
        f(*args)
        print("OK   ", f.__name__)
    except Exception as e:
        print("FAIL ", f.__name__, "--", e)


coins = data.coins()
mask = coins > 100

with tempfile.TemporaryDirectory() as tmp:
    store = icechunk.Repository.create(icechunk.local_filesystem_storage(tmp)).writable_session("main").store
    z = zarr.create_array(store, name="coins", data=coins)
    zmask = zarr.create_array(store, name="mask", data=mask)

    run(filters.gaussian, z, 2)
    run(measure.label, zmask)
    run(transform.resize, z, (100, 100))
    run(filters.threshold_otsu, z)
    run(morphology.remove_small_objects, zmask, 50)
    run(segmentation.watershed, z, measure.label(mask))
