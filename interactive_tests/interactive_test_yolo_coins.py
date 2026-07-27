"""Two class-agnostic detectors on the coins image, drawn in napari.

Informal and meant to be played with -- change the thresholds and watch the
boxes multiply or vanish. Coins are the point of the test: they are obviously
objects and belong to no COCO class, so a detector that finds them is doing
the job a segmenter needs, and one that classifies would find nothing here.

The two detectors are the same kind of model doing the same job:

  fastsam            ultralytics' FastSAM, in the shared 'pytorch' env
  object_aware_yolo  MobileSAMv2's fine-tuned YOLO, via segment-everything

Both return boxes in skop's canonical row-major order, which is also napari's,
so drawing them is a reshape and nothing more.

Run it with napari available:

    uv run --with napari --with pyqt5 python interactive_tests/interactive_test_yolo_coins.py

The first run builds both environments -- two PyTorch stacks, several
gigabytes, minutes each -- and then each model downloads its own weights.
Build phases are printed as they happen. If one detector fails for any reason
the other still runs and still gets drawn, so this stays useful with one
environment working.
"""

import time

import napari
import numpy as np
from skimage.data import coins

import skop
from skop import boxes
from skop.ops.detect import fastsam, object_aware_yolo

# Lower to find more, and more spuriously. Raise if the boxes look like
# confetti. iou is the one that matters most for touching objects: at 0.9 two
# detections overlapping by less than 90% are kept as separate things.
CONF = 0.4
IOU = 0.9

image = coins()
print(f"coins: {image.shape} {image.dtype}, range {image.min()}-{image.max()}")

runner = skop.Runner()

# Building a PyTorch environment takes minutes with nothing else to show for
# it, so say which phase it is in rather than appearing to hang.
_phase = None


def on_build(title, current, maximum):
    global _phase
    if title != _phase:
        _phase = title
        print(f"  [build] {title}")


runner.subscribe_build_progress(on_build)


def detect(op, name):
    """Run one detector, or explain why it could not run."""
    print(f"\nRunning {name}...")
    started = time.perf_counter()
    try:
        result = runner.run(
            op,
            image=image,
            conf=CONF,
            iou=IOU,
            on_progress=lambda event: print(f"  {event.message}"),
        )
    except Exception as exc:  # noqa: BLE001 -- an interactive script, not a test
        # A missing environment, a failed build, a model that will not load:
        # report it and let the other detector still run.
        print(f"  failed: {type(exc).__name__}: {exc}")
        return None

    elapsed = time.perf_counter() - started
    print(f"  {len(result.boxes)} boxes in {elapsed:.1f}s")
    if len(result.boxes):
        heights = result.boxes[:, 2] - result.boxes[:, 0]
        widths = result.boxes[:, 3] - result.boxes[:, 1]
        print(
            f"  sizes {heights.min():.0f}-{heights.max():.0f} tall, "
            f"{widths.min():.0f}-{widths.max():.0f} wide"
        )
    return result


fast = detect(fastsam, "fastsam")
aware = detect(object_aware_yolo, "object_aware_yolo")

if fast is None and aware is None:
    raise SystemExit("\nBoth detectors failed -- see the messages above.")

print("\nOpening napari...")
viewer = napari.Viewer()
viewer.add_image(image, name="coins")

# An empty layer still gets added, so the viewer looks the same either way and
# a detector that found nothing is visible as an empty layer rather than a
# missing one.
for result, name, color in [
    (fast, "fastsam", "yellow"),
    (aware, "object aware", "cyan"),
]:
    found = result.boxes if result is not None else np.zeros((0, 4))
    viewer.add_shapes(
        boxes.to_napari(found),
        shape_type="rectangle",
        edge_color=color,
        edge_width=2,
        face_color="transparent",
        name=f"{name} ({len(found)})",
    )

napari.run()
