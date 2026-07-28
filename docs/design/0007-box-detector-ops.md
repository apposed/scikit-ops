# 0007 — Bounding box detector ops

**Status:** implemented. `skop.ops.detect` holds both detectors,
`envs/segment-everything` and `envs/pytorch` are the two environments,
`skop.boxes` holds the converters, and `BoxesData` names the canonical format
in `skop.types`. Scores did not survive contact — see
[per-object-features](../spec/per-object-features.md). Still unwritten: the
classical `regionprops` detector, Faster R-CNN, labels-to-boxes, and the
`Choices` list that would let a workflow pick between them, which belongs to
[workflow-ops](../spec/workflow-ops.md).

## The kind

A box detector takes an image and returns boxes:

```python
@op(env=...)
def some_detector(image: ImageData, ...) -> Boxes   # NamedTuple(boxes, scores)
```

This is stage one of the segmenter in
[workflow-ops](../spec/workflow-ops.md), feeding a mask detector
([0008](0008-mask-detector-ops.md)). There are many possible
implementations — FastSAM, MobileSAMv2's `ObjectAwareModel`, Faster R-CNN, a
classical threshold-and-`regionprops` detector, boxes from an existing label
image — and the point of the kind is that the segmenter workflow can hold a
`Choices` list over them.

**These detectors are class-agnostic on purpose.** What the segmenter wants is
"where are the objects", not "which of 80 COCO classes is this". A
COCO-pretrained detector has no class for a cell or a coin and returns nothing;
the useful family is the single-class "find everything" detectors — FastSAM,
`ObjectAwareModel`, and a threshold-based detector, all of which answer the
question actually being asked. That is also why classes are absent from the
return type below.

Since they must be substitutable, they share a signature (workflow-ops,
consideration 5). The shared core is `image`, plus `conf`, `iou` and
`max_det` — the knobs users actually turn, and they mean the same thing
everywhere.
Implementation-specific extras (`imgsz`) get defaults and stay at the end. A
classical detector would have to interpret `conf`/`iou` loosely or ignore them;
that is fine, and a good early test of how strict "shared signature" has to be.

**Scores come back with the boxes.** A `NamedTuple` of `boxes` and `scores`
(0001) — the workflow filters, and a front end can show confidence. No classes:
a class-agnostic detector has nothing to put there.

## What we do and do not port

The rule this project exists for: **we wrap, we don't migrate code trees.**
Porting fairly contained Python — Richardson-Lucy, a Gaussian PSF — is fine.
Anything with a vendored dependency tree gets a pixi environment and a thin
wrapper instead. That is the whole motivation: an op is a few lines of glue over
a library that lives in an environment we do not have to reconcile with anything
else.

So:

- **Port:** the bounding box converters, reimplemented as utilities (below).
  Four numbers transposed — not worth a dependency.
- **Do not port:** everything else in segment-everything. It is on PyPI
  (`segment-everything`, currently 0.3), so it becomes a dependency of one
  environment, and the op is a light wrapper over `YoloDetector`.

### Why the vendored tree makes this obvious

`segment-everything` vendors MobileSAMv2 from
[ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM/tree/master/MobileSAMv2)
— all of `mobilesamv2`, `efficientvit`, `tinyvit`, an ultralytics 8.0.120 fork,
and the `PromptGuidedDecoder` weights, landed in one commit of 132 files and
28,847 lines. It is not a clean upstream copy that could be re-vendored by
download. The tweaks:

1. `ObjectAwareModel` was **moved**. Upstream it is `mobilesamv2.promt_mobilesamv2`
   (missing `p`); in segment-everything it is
   `vendored/object_detection/ultralytics/prompt_mobilesamv2`, spelling fixed and
   re-rooted under the fork where its `YOLO` base class lives.
2. The ultralytics fork was **trimmed** — upstream `assets`, `hub`, `models`,
   `tracker`, `vit` are gone; `nn` and `yolo` remain.
3. `ultralytics/__init__.py` was **gutted** as a consequence, down to
   `__version__ = "8.0.120"`.
4. Entry-point imports were **rewritten to relative** (`from ..yolo.cfg import
   get_cfg`) so the fork imports as a nested package — but only at the entry
   point. Fourteen files deeper down still use absolute `from ultralytics...`,
   which is the real reason `get_object_aware_model` must
   `sys.path.insert(0, vendored/object_detection)`. The docstring blames the
   pickled weights; the unrewritten imports need it independently.
5. Later, `weights_only=False` was patched across four vendored files for
   torch compatibility (`fb55fa0`).

Re-doing that here would mean owning a fork of a fork. Wrapping it costs one
`pixi.toml`.

It also means the environment must contain exactly one `ultralytics`: once that
`sys.path.insert` runs, a plain `import ultralytics` resolves to the vendored
8.0.120 fork. Which is the second reason for two environments rather than one.

## Two detectors, two environments

**`envs/segment-everything`** — a dedicated environment pinning
`segment-everything` from PyPI, with whatever torch it needs and nothing else it
could shadow. The op is a wrapper:

```python
@op(env="segment-everything")
def object_aware_yolo(image: ImageData, conf=0.4, iou=0.9, max_det=400) -> Boxes:
    from segment_everything.object_detectors.yolo_detector import YoloDetector
    from segment_everything.weights_helper import get_weights_path

    ...
```

`ObjectAwareModel` is the detector that actually works on cells, so this is the
one we need first even though it is the awkward one. Note that
`weights_helper.get_weights_path` fetches through gdown into
`~/.cache/segment_everything` — its own cache, outside `skop.assets`. Leave it;
touching it means patching a dependency. The HuggingFace mirror already in
`WEIGHTS_URLS` is there if gdown breaks.

**`envs/pytorch`** — and this is the part worth getting right. A *good* PyTorch
environment: a well-maintained, well-used ultralytics, and over time Cellpose and
the other torch-based libraries that are happy to coexist. One build, one warm
worker, shared by many ops (0002). `envs/cellpose` is its starting point — it
already has the per-platform torch split — and Cellpose should move into it once
there is a second occupant.

The op that lands there is **FastSAM**, which ultralytics ships and maintains
(`from ultralytics import FastSAM`). It is a YOLOv8-seg trained on SA-1B and, in
ultralytics' own words, "will recognize and segment all objects as the same
class" — the same job as `ObjectAwareModel`, from a package that is actively
maintained and a plain conda/pip install. If it disappoints, YOLOE's prompt-free
variant is the fallback, though its 1200-category vocabulary makes it open-set
rather than truly class-agnostic.

This is the environment we want ops to land in by default. The
segment-everything env is the exception we accept for one model we cannot get
any other way; the pytorch env is the road.

**The road checks out.** `ultralytics + cellpose>4 + micro_sam>1.6 +
pytorch-gpu` resolves from conda-forge alone — no PyPI packages at all — on
python 3.11, torch 2.7.1, cuda 12.9. So the intended end state, one torch
environment holding the detector, Cellpose and the mask detector of 0008, is
not wishful: it solves today. They stay out of `envs/pytorch` only until their
ops move there.

The contrast with the environment next door is the whole argument. Everything
above comes from conda; `envs/segment-everything` needs its dependencies
restated as conda packages before it will solve at all, because left to PyPI
uv resolves `torchvision` against the conda-pinned `torch` and finds no
version that matches. That recipe is not something a solver finds — it was
arrived at by trial and error in
`../napari-ai-lab/pixi/microsam_cellposesam_czi` and copied here. A wrapped
dependency costs one pixi.toml, but not always an easy one.

Both detectors return the same thing, so the segmenter workflow can offer both
in one `Choices` list — the interesting comparison being exactly whether the
fine-tuned `ObjectAwareModel` beats a maintained FastSAM on cells. They are the
same kind of model doing the same job, which is what makes it a fair question.

FastSAM also returns masks, so it could later serve as a mask detector and skip
the second stage entirely. Out of scope here, but it is the reason not to bury
`results[0].masks` too deep.

## Bounding box formats

The one thing we do reimplement. Boxes in play:

| Source | Layout |
| --- | --- |
| YOLO / most detectors | `[x1, y1, x2, y2]` |
| micro_sam | `[y1, x1, y2, x2]` |
| napari rectangle | `[[y1, x1], [y2, x2]]` (corners) |
| skimage `regionprops.bbox` | `(min_row, min_col, max_row, max_col)` |
| YOLO training labels | `[xc, yc, w, h]`, normalized |

segment-everything handles this with `convert_yolo_boxes_to_microsam` and
`convert_yolo_boxes_to_napari` at the bottom of `yolo_detector.py` — attached to
whichever detector happened to need them, which is why a second detector would
copy them.

Two proposals:

1. **One canonical format, converted at the edges.** Row-major `(N, 4)` as
   `[min_y, min_x, max_y, max_x]`, matching `PointsData`'s rule that coordinates
   are in the axis order of the image they came from — and matching micro_sam
   already. Every detector op converts on the way out; a front end converts on
   the way in; nothing in the middle converts. Name it with a `BoxesData` alias
   in `skop.types` over the existing `Role.shapes`.

2. **Converters are utilities, not ops.** Making them ops would buy a
   shared-memory round trip and a worker. Put them in a plain module —
   `src/skop/boxes.py` — that both the host and any worker can import, on the
   condition that it stays numpy-and-stdlib only, like `skop.assets` and
   `skop._progress`. `skop/ops/_util.py` is the alternative home, but ops-only is
   the wrong scope: a workflow op and a napari layer both need these.

## Other detectors, later

Faster R-CNN (trainable, torchvision — belongs in the pytorch env), a classical
threshold-and-`regionprops` detector, and labels-to-boxes. The classical one is
worth writing early despite being the least interesting: no GPU, no weights,
runs in the `skimage` env, and it makes the whole two-stage workflow testable on
a laptop.
