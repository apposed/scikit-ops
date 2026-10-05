"""The two big 2D images the yolo-*.py scripts share

Not a script; the three yolo-*.py scripts import it from beside them.

Each image is kept as test_images/yolo-<name>.zarr, a zarr group holding one
plain 2D array per pyramid level -- y,x for EM, y,x,rgb for aerial -- named
by the source's level number, finest first. Every level halves the one
before, so level k is 2**k smaller than level 0.

    em      IDR idr0083A/9822152, serial-section EM, uint16. Levels 2-10 of
            the 11; level 0 (93184 x 144384) and 1 stay on the IDR.
    aerial  "hatirjheel_partial", a drone mosaic of Dhaka from
            OpenAerialMap, 3 cm per pixel, uint8 RGB, levels 0-6 from the
            GeoTIFF's own overviews. GPAD, CC-BY 4.0.

The same object is big at one level and small at another, which is the point:
a detector finds what is near the size it was trained at (yolo-sam-tiling.md).
"""

from pathlib import Path

IMAGES = Path(__file__).resolve().parents[3] / "test_images"

DATA = {
    "em": {
        "source": "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr",
        "levels": list(range(2, 11)),
        "credit": "IDR idr0083A/9822152",
    },
    "aerial": {
        "source": (
            "https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/"
            "6aa12cc6c0980283f7857e64/0/6aa12cc6c0980283f7857e65.tif"
        ),
        "credit": "hatirjheel_partial, GPAD, CC-BY 4.0, via OpenAerialMap",
    },
}


def path(name: str) -> Path:
    return IMAGES / f"yolo-{name}.zarr"


def open_levels(name: str) -> dict:
    """Level number -> zarr array, finest first. Nothing is read yet."""
    import zarr

    p = path(name)
    if not p.exists():
        raise SystemExit(f"No {p}; run yolo-download.py {name} first")
    group = zarr.open_group(p, mode="r")
    return {k: group[str(k)] for k in group.attrs["levels"]}
