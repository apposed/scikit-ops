# 0013 — Plan to add bounding box detector ops

**Status:** implemented. This is the build plan that produced
[0007](0007-box-detector-ops.md) and [0008](0008-mask-detector-ops.md); those
two say what the code ended up being, and this says what the plan was. Kept
rather than deleted because the ordering and the decisions it assumed are not
recorded anywhere else.

Implements [design 0007](0007-box-detector-ops.md). Two detectors, two
environments, one box format.

## Decisions this plan assumes

- **Namespace `src/skop/ops/detect/`**, a package with one module per op —
  matching the verb namespaces already there (`segment`, `threshold`, `generate`,
  `deconvolve`). A package rather than a module because the two ops live in
  different environments.
- **Both ops return the same `NamedTuple`.**

  ```python
  class Boxes(NamedTuple):
      boxes: BoxesData  # (N, 4) float32, [min_y, min_x, max_y, max_x]
      scores: np.ndarray  # (N,) float32
  ```

  Field names become output names (0001). Classes are left out until a detector
  means something by them.
- **Canonical box format is row-major**, `[min_y, min_x, max_y, max_x]`, per
  0007. Every op converts on the way out; nothing in the middle converts.
- **Converters live in `src/skop/boxes.py`**, numpy-and-stdlib only, importable
  from host and worker alike.
- **`envs/pytorch` is a shared environment from day one**, not a YOLO
  environment that later grows. Cellpose moves into it in a follow-up; this plan
  only creates it and puts one op in it.

## What gets written

```
src/skop/boxes.py                              converters, host + worker safe
src/skop/types.py                              add BoxesData
src/skop/ops/_util.py                          add to_rgb
src/skop/ops/detect/__init__.py                re-exports both ops
src/skop/ops/detect/fastsam.py                 fastsam            (env pytorch)
src/skop/ops/detect/object_aware_yolo.py       object_aware_yolo  (env segment-everything)
envs/pytorch/pixi.toml                         new: torch + ultralytics
envs/segment-everything/pixi.toml              new: torch + segment-everything
tests/test_boxes.py                            host-only, no env needed
tests/test_ops.py                              spec-level assertions
tests/test_ops_e2e.py                          real runs, both envs
interactive_tests/interactive_test_yolo_coins.py
README.md                                      add ops and envs to the table
```

## Steps

**1. `skop/boxes.py` and `BoxesData`.** The whole of it is array reshaping, so it
lands first and is testable on any machine with no environment built:

```python
def from_xyxy(boxes) -> np.ndarray      # detector order -> canonical
def to_xyxy(boxes) -> np.ndarray
def to_napari(boxes) -> np.ndarray      # (N, 2, 2) corner pairs for a Shapes layer
def from_napari(boxes) -> np.ndarray    # accepts (N, 2, 2) and (N, 4, 2)
def from_labels(labels) -> np.ndarray   # regionprops-style bboxes of a label image
```

Take and return `(N, 4)` float32 arrays, not lists — segment-everything's
converters build Python lists in a loop, and the ops hand these straight to the
codec. Empty input must give `(0, 4)`, not `(0,)`; a detector finding nothing is
the normal case and everything downstream indexes columns.

`BoxesData = Annotated[np.ndarray, Role.shapes]` goes in `skop.types` beside the
existing aliases. `Role.shapes` already exists.

**2. `envs/pytorch/pixi.toml`.** Copy the per-platform torch split from
`envs/cellpose` verbatim — CUDA builds on linux-64/win-64, plain pytorch on
osx-arm64 — and add `ultralytics`. Two things to check while writing it:

- Whether `ultralytics` is on conda-forge at a current version. If not it goes in
  `[pypi-dependencies]`, which is fine but should be a deliberate note in the
  file, not a silent choice.
- Ultralytics writes a settings file and downloads weights to a directory of its
  own choosing, which in a worker process may be the repository root. Set
  `YOLO_CONFIG_DIR` in the env, or fetch weights through `skop.assets` and pass
  an absolute path. Decide this while writing step 3 — it is the difference
  between a clean cache and `.pt` files appearing in the working directory.

Name it `skop-pytorch` in `[workspace]`, following the others.

**3. `src/skop/ops/detect/fastsam.py` — the class-agnostic detector.**

Not a COCO-pretrained YOLO. A stock detector classifies into 80 categories, none
of which is a cell or a coin, and on this kind of image it returns nothing. What
the segmenter needs is "where are the objects", so the op is **FastSAM**: a
YOLOv8-seg trained on SA-1B that, per ultralytics, "will recognize and segment
all objects as the same class". Shipped and maintained by ultralytics, plain
install, no vendoring.

```python
class PretrainedModel(Enum):
    small = "FastSAM-s.pt"
    large = "FastSAM-x.pt"

@op(env="pytorch")
def fastsam(image: ImageData, model: PretrainedModel = PretrainedModel.small,
            conf: float = 0.4, iou: float = 0.9, max_det: int = 300,
            imgsz: int = 1024) -> Boxes:
```

`from ultralytics import FastSAM` in the body. Follows `stardist2d.py`: an `Enum`
of known models so a front end gets a combo box, sliders on `conf`/`iou` through
the UI-hint dicts, weights downloaded on first use with a note in the docstring.
The `conf=0.4, iou=0.9, imgsz=1024` defaults are ultralytics' own for FastSAM and
happen to match what `object_aware_yolo` uses, which keeps the two ops
substitutable in practice and not just in signature.

`retina_masks=True` and the mask output are deliberately ignored here — this op
returns boxes. Keep `results[0].masks` reachable in the code rather than buried;
0007 notes FastSAM may later serve as a mask detector too.

If FastSAM turns out to be weak on cells, the fallback is YOLOE's prompt-free
variant (`yoloe-11s-seg-pf.pt`), which is open-set over ~1200 categories rather
than truly class-agnostic. Same op shape, one import changed.

`to_rgb` in `ops/_util.py` is the other half of the existing `to_gray`:
2-D or single-channel input becomes `(H, W, 3)` uint8, percentile-normalized.
Every detector here needs it, and getting it wrong is the most likely cause of
"the model finds nothing".

Result extraction is `results[0].boxes.xyxy.cpu().numpy()` and `.conf`, then
`boxes.from_xyxy`.

**4. `envs/segment-everything/pixi.toml`.** `segment-everything = "==0.3"` from
PyPI, python 3.11, appose, and torch — and deliberately **no** `ultralytics`.
The package vendors its own ultralytics 8.0.120 fork and puts it on `sys.path`;
a second one in the same environment is the failure mode 0007 describes. That
exclusion needs a comment in the file, because it looks like an oversight.

`opencv-python` comes in as a dependency and `YoloDetector.get_results` calls
`cv2.cvtColor` — leave it, it is the dependency's business.

**5. `src/skop/ops/detect/object_aware_yolo.py`** — a thin wrapper, which is the
whole point:

```python
@op(env="segment-everything")
def object_aware_yolo(
    image: ImageData,
    conf: float = 0.4,
    iou: float = 0.9,
    max_det: int = 400,
    imgsz: int = 1024,
) -> Boxes:
    from segment_everything.object_detectors.yolo_detector import YoloDetector
    from segment_everything.weights_helper import get_device, get_weights_path

    ...
```

Notes:

- Model type string is `"ObjectAwareModelFromMobileSamV2"`; weights come from
  `get_weights_path("ObjectAwareModel")`. Prefer the `ObjectAwareModelHuggingFace`
  entry if the gdown fetch proves flaky — both are already in `WEIGHTS_URLS`.
- Weights land in `~/.cache/segment_everything`, not the skop asset cache. Leave
  it; changing it means patching a dependency.
- `get_device()` handles cuda/mps/cpu. Use it rather than reimplementing.
- `get_bounding_boxes` returns `xyxy` and drops scores. Getting scores means
  calling `get_results` and reading `obj_results[0].boxes.conf` — do that, so
  both ops fill the same `NamedTuple`.
- Construct the detector inside the op. It is a per-call model load, which is
  slow and correct; caching a model across calls in a warm worker is a separate
  design question (it applies to every op here, not just this one).

**6. Tests.**

- `tests/test_boxes.py` — host-only, no marker. Round trips through each
  converter, `(0, 4)` on empty, `from_labels` against a hand-built label image,
  and the specific assertion that `from_xyxy` of `[10, 20, 30, 40]` is
  `[20, 10, 40, 30]`. That transposition is the bug this whole module exists to
  stop happening twice.
- `tests/test_ops.py` — both ops discovered, envs are `pytorch` and
  `segment-everything`, outputs are `("boxes", "scores")`, the `boxes` output
  carries `Role.shapes`, and bump the `len(SPECS) >= 15` floor.
- `tests/test_ops_e2e.py` — `@pytest.mark.env("pytorch")` and
  `@pytest.mark.env("segment-everything")`. Assert shape `(N, 4)`, dtype, that
  `min_y < max_y` and `min_x < max_x` for every row, and that boxes lie inside
  the image. Because both detectors are class-agnostic, `N > 0` on a synthetic
  blobs image is a legitimate assertion — that is the one that catches a broken
  `to_rgb` or a silently mis-set threshold. Both env markers on one test for a
  cross-detector comparison, which will skip on most machines and that is fine.
- These are CPU-runnable, so no `gpu` marker — but the first run of either
  environment is a multi-gigabyte download, which is exactly what `--build-envs`
  is for.

**7. `interactive_tests/interactive_test_yolo_coins.py`.** Flat script, no
pytest, prints as it goes — the house style:

```python
from skimage.data import coins

image = coins()  # (303, 384) uint8

runner = skop.Runner()
a = runner.run(fastsam, image=image)
b = runner.run(object_aware_yolo, image=image)
print(f"fastsam: {len(a.boxes)} boxes, object aware: {len(b.boxes)} boxes")

viewer = napari.Viewer()
viewer.add_image(image, name="coins")
viewer.add_shapes(
    boxes.to_napari(a.boxes),
    shape_type="rectangle",
    edge_color="yellow",
    face_color="transparent",
    name="fastsam",
)
viewer.add_shapes(
    boxes.to_napari(b.boxes),
    shape_type="rectangle",
    edge_color="cyan",
    face_color="transparent",
    name="object aware",
)
napari.run()
```

Napari's `rectangle` accepts a `(2, 2)` pair of opposite corners, which is what
`to_napari` emits, and its row-major axis order is why the canonical format is
row-major — no conversion in the viewer.

`coins` is the right test image precisely because both detectors are
class-agnostic: coins on a dark background are obviously objects and belong to no
COCO class, so finding them is evidence the detector is doing the job the
segmenter needs. Both layers should light up, and the interesting thing to look
at is where they disagree.

Three things to get right:

- Print the box count per detector before opening the viewer, so a zero is
  visible in the terminal rather than inferred from an empty layer.
- Wrap each detector in try/except so the script still runs with one environment
  built. Same pattern as `interactive_test_decon.py` does for cupy.
- Empty results must still add a Shapes layer (napari accepts an empty list), so
  the viewer looks the same either way.

Run it the same way as the others:
`uv run --with napari --with pyqt5 python interactive_tests/interactive_test_yolo_coins.py`

**8. README.** Add both ops and both environments to the table.

## Sequencing

Step 1 and its tests are one unit and land alone — no environment, no download,
and everything after depends on the format being fixed. Steps 2–3 are the second
unit and the one worth doing carefully, since `envs/pytorch` is meant to outlive
this feature. Steps 4–5 are the third and are mostly plumbing. Step 7 needs both
environments built, so it closes the work.

## Not in scope

Faster R-CNN, the classical `regionprops` detector, labels-to-boxes as an op
(the utility in step 1 covers the need for now), moving Cellpose into
`envs/pytorch`, the segmenter workflow that consumes these (0005), and caching a
loaded model across calls in a warm worker.
