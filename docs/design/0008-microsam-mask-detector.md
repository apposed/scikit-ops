# 0008 — A micro_sam mask detector op

**Status:** scoping a port. Nothing implemented. The point here is to work out
what has to come across and where the pieces land, not to fix an API.

## What we want

A mask detector: given an image and a set of bounding boxes, return one mask per
box. This is the second stage of the segmenter in [0005](0005-workflow-ops.md);
the first stage is a bounding box detector op ([0007](0007-box-detector-ops.md)).
Because the two stages live in different environments, the thing that joins them
is a workflow op.

Port it, don't depend on it. `../segment-everything` carries a vendored
MobileSAMv2 tree, a `BaseDetector` class hierarchy, and `StackedLabels`; we want
none of that. The code we actually need is
`segment_everything/mask_detectors/microsam.py`, which is about thirty lines.

## What the op needs

From micro_sam, both imported in the function body (0001):

- `micro_sam.sam_annotator._state.AnnotatorState` — used to build the predictor
  and precompute image embeddings. This is a private module belonging to the
  napari annotator, and it is a singleton with `reset_state()`; we should check
  whether `micro_sam.util` exposes a public predictor + embeddings path and use
  that instead. Worth doing before the port, not after.
- `micro_sam.prompt_based_segmentation.segment_from_box(predictor, box,
  image_embeddings=...)` — one call per box, returning a boolean mask.

Nothing else: the rest of the file is numpy.

The shape of the op is roughly

```python
@op(env="microsam")
def segment_from_boxes(image: ImageData, boxes: BoxesData, model_type: str = "vit_b_lm") -> ?
```

with two things to settle:

- **Embeddings are the expensive part** and they depend only on the image. One
  op call per image, looping over boxes inside, keeps that cost paid once — and
  avoids a round trip per box (0005, "round trips cost").
- **What comes back.** micro_sam gives a boolean mask per box, and SAM masks may
  overlap, so a label image loses information. Options are a `(N, Y, X)` boolean
  stack, or labels plus the accepted boxes. The original also drops empty
  predictions and reports `area`; a role-annotated array can't carry that, so
  either it goes in a `NamedTuple` or it goes away. Undecided.

## Environment

First choice is the shared `envs/pytorch` environment proposed in
[0007](0007-box-detector-ops.md) — micro_sam is a normal conda package and is
exactly the kind of occupant that environment is for. Check whether it drags
napari in (heavy, but harmless inside a worker) and whether it can be solved
alongside ultralytics and Cellpose; only if it can't does it get its own
`envs/microsam`. Weights are micro_sam's own download, in its own cache, so
`skop.assets` isn't involved.

## Bounding box format

Box layouts and where the converters live are settled in
[0007](0007-box-detector-ops.md). The relevant point here: micro_sam wants
`[y1, x1, y2, x2]`, which is the canonical row-major format proposed there, so
this op should need no conversion at all. `segment-everything` needs
`convert_yolo_boxes_to_microsam` only because its detectors hand out `xyxy`.

## Not porting

`BaseDetector` / `BaseMaskDetector` (ops are functions — 0001), the vendored
model trees, and `StackedLabels`. If overlapping masks need a container later,
that is its own design.
